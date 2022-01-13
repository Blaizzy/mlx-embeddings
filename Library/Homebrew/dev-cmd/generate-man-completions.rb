# typed: false
# frozen_string_literal: true

require "formula"
require "erb"
require "ostruct"
require "cli/parser"
require "completions"

module Homebrew
  extend T::Sig

  module_function

  SOURCE_PATH = (HOMEBREW_LIBRARY_PATH/"manpages").freeze
  TARGET_MAN_PATH = (HOMEBREW_REPOSITORY/"manpages").freeze
  TARGET_DOC_PATH = (HOMEBREW_REPOSITORY/"docs").freeze

  sig { returns(CLI::Parser) }
  def generate_man_completions_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Generate Homebrew's manpages and shell completions.
      EOS
      switch "--fail-if-not-changed",
             description: "Return a failing status code if no changes are detected in the manpage outputs. "\
                          "This can be used to notify CI when the manpages are out of date. Additionally, "\
                          "the date used in new manpages will match those in the existing manpages (to allow "\
                          "comparison without factoring in the date)."
      named_args :none
    end
  end

  def generate_man_completions
    args = generate_man_completions_args.parse

    Commands.rebuild_internal_commands_completion_list
    regenerate_man_pages(preserve_date: args.fail_if_not_changed?, quiet: args.quiet?)
    Completions.update_shell_completions!

    diff = system_command "git", args: [
      "-C", HOMEBREW_REPOSITORY, "diff", "--exit-code", "docs/Manpage.md", "manpages", "completions"
    ]

    return unless diff.status.success?

    puts "No changes to manpage or completions output detected."
    Homebrew.failed = true if args.fail_if_not_changed?
  end

  def regenerate_man_pages(preserve_date:, quiet:)
    Homebrew.install_bundler_gems!

    markup = build_man_page(quiet: quiet)
    convert_man_page(markup, TARGET_DOC_PATH/"Manpage.md", preserve_date: preserve_date)
    markup = I18n.transliterate(markup, locale: :en)
    convert_man_page(markup, TARGET_MAN_PATH/"brew.1", preserve_date: preserve_date)
  end

  def build_man_page(quiet:)
    template = (SOURCE_PATH/"brew.1.md.erb").read
    variables = OpenStruct.new

    variables[:commands] = generate_cmd_manpages(Commands.internal_commands_paths)
    variables[:developer_commands] = generate_cmd_manpages(Commands.internal_developer_commands_paths)
    variables[:official_external_commands] =
      generate_cmd_manpages(Commands.official_external_commands_paths(quiet: quiet))
    variables[:global_cask_options] = global_cask_options_manpage
    variables[:global_options] = global_options_manpage
    variables[:environment_variables] = env_vars_manpage

    readme = HOMEBREW_REPOSITORY/"README.md"
    variables[:lead] =
      readme.read[/(Homebrew's \[Project Leader.*\.)/, 1]
            .gsub(/\[([^\]]+)\]\([^)]+\)/, '\1')
    variables[:plc] =
      readme.read[/(Homebrew's \[Project Leadership Committee.*\.)/, 1]
            .gsub(/\[([^\]]+)\]\([^)]+\)/, '\1')
    variables[:tsc] =
      readme.read[/(Homebrew's \[Technical Steering Committee.*\.)/, 1]
            .gsub(/\[([^\]]+)\]\([^)]+\)/, '\1')
    variables[:maintainers] =
      readme.read[/(Homebrew's other current maintainers .*\.)/, 1]
            .gsub(/\[([^\]]+)\]\([^)]+\)/, '\1')
    variables[:alumni] =
      readme.read[/(Former maintainers .*\.)/, 1]
            .gsub(/\[([^\]]+)\]\([^)]+\)/, '\1')

    ERB.new(template, trim_mode: ">").result(variables.instance_eval { binding })
  end

  def sort_key_for_path(path)
    # Options after regular commands (`~` comes after `z` in ASCII table).
    path.basename.to_s.sub(/\.(rb|sh)$/, "").sub(/^--/, "~~")
  end

  def convert_man_page(markup, target, preserve_date:)
    manual = target.basename(".1")
    organisation = "Homebrew"

    # Set the manpage date to the existing one if we're checking for changes.
    # This avoids the only change being e.g. a new date.
    date = if preserve_date && target.extname == ".1" && target.exist?
      /"(\d{1,2})" "([A-Z][a-z]+) (\d{4})" "#{organisation}" "#{manual}"/ =~ target.read
      Date.parse("#{Regexp.last_match(1)} #{Regexp.last_match(2)} #{Regexp.last_match(3)}")
    else
      Date.today
    end
    date = date.strftime("%Y-%m-%d")

    shared_args = %W[
      --pipe
      --organization=#{organisation}
      --manual=#{target.basename(".1")}
      --date=#{date}
    ]

    format_flag, format_desc = target_path_to_format(target)

    puts "Writing #{format_desc} to #{target}"
    Utils.popen(["ronn", format_flag] + shared_args, "rb+") do |ronn|
      ronn.write markup
      ronn.close_write
      ronn_output = ronn.read
      odie "Got no output from ronn!" if ronn_output.blank?
      case format_flag
      when "--markdown"
        ronn_output = ronn_output.gsub(%r{<var>(.*?)</var>}, "*`\\1`*")
                                 .gsub(/\n\n\n+/, "\n\n")
                                 .gsub(/^(- `[^`]+`):/, "\\1") # drop trailing colons from definition lists
                                 .gsub(/(?<=\n\n)([\[`].+):\n/, "\\1\n<br>") # replace colons with <br> on subcommands
      when "--roff"
        ronn_output = ronn_output.gsub(%r{<code>(.*?)</code>}, "\\fB\\1\\fR")
                                 .gsub(%r{<var>(.*?)</var>}, "\\fI\\1\\fR")
                                 .gsub(/(^\[?\\fB.+): /, "\\1\n    ")
      end
      target.atomic_write ronn_output
    end
  end

  def target_path_to_format(target)
    case target.basename
    when /\.md$/    then ["--markdown", "markdown"]
    when /\.\d$/    then ["--roff", "man page"]
    else
      odie "Failed to infer output format from '#{target.basename}'."
    end
  end

  def generate_cmd_manpages(cmd_paths)
    man_page_lines = []

    # preserve existing manpage order
    cmd_paths.sort_by(&method(:sort_key_for_path))
             .each do |cmd_path|
      cmd_man_page_lines = if (cmd_parser = CLI::Parser.from_cmd_path(cmd_path))
        next if cmd_parser.hide_from_man_page

        cmd_parser_manpage_lines(cmd_parser).join
      else
        cmd_comment_manpage_lines(cmd_path)
      end

      man_page_lines << cmd_man_page_lines
    end

    man_page_lines.compact.join("\n")
  end

  def cmd_parser_manpage_lines(cmd_parser)
    lines = [format_usage_banner(cmd_parser.usage_banner_text)]
    lines += cmd_parser.processed_options.map do |short, long, _, desc, hidden|
      next if hidden

      if long.present?
        next if Homebrew::CLI::Parser.global_options.include?([short, long, desc])
        next if Homebrew::CLI::Parser.global_cask_options.any? do |_, option, description:, **|
                  [long, "#{long}="].include?(option) && description == desc
                end
      end

      generate_option_doc(short, long, desc)
    end.compact
    lines
  end

  def cmd_comment_manpage_lines(cmd_path)
    comment_lines = cmd_path.read.lines.grep(/^#:/)
    return if comment_lines.empty?
    return if comment_lines.first.include?("@hide_from_man_page")

    lines = [format_usage_banner(comment_lines.first).chomp]
    comment_lines.slice(1..-1)
                 .each do |line|
      line = line.slice(4..-2)
      unless line
        lines.last << "\n"
        next
      end

      # Omit the common global_options documented separately in the man page.
      next if line.match?(/--(debug|help|quiet|verbose) /)

      # Format one option or a comma-separated pair of short and long options.
      lines << line.gsub(/^ +(-+[a-z-]+), (-+[a-z-]+) +/, "* `\\1`, `\\2`:\n  ")
                   .gsub(/^ +(-+[a-z-]+) +/, "* `\\1`:\n  ")
    end
    lines.last << "\n"
    lines
  end

  sig { returns(String) }
  def global_cask_options_manpage
    lines = ["These options are applicable to the `install`, `reinstall`, and `upgrade` " \
             "subcommands with the `--cask` flag.\n"]
    lines += Homebrew::CLI::Parser.global_cask_options.map do |_, long, description:, **|
      generate_option_doc(nil, long.chomp("="), description)
    end
    lines.join("\n")
  end

  sig { returns(String) }
  def global_options_manpage
    lines = ["These options are applicable across multiple subcommands.\n"]
    lines += Homebrew::CLI::Parser.global_options.map do |short, long, desc|
      generate_option_doc(short, long, desc)
    end
    lines.join("\n")
  end

  sig { returns(String) }
  def env_vars_manpage
    lines = Homebrew::EnvConfig::ENVS.flat_map do |env, hash|
      entry = "- `#{env}`:\n  <br>#{hash[:description]}\n"
      default = hash[:default_text]
      default ||= "`#{hash[:default]}`." if hash[:default]
      entry += "\n\n  *Default:* #{default}\n" if default

      entry
    end
    lines.join("\n")
  end

  def generate_option_doc(short, long, desc)
    comma = (short && long) ? ", " : ""
    <<~EOS
      * #{format_short_opt(short)}#{comma}#{format_long_opt(long)}:
        #{desc}
    EOS
  end

  def format_short_opt(opt)
    "`#{opt}`" unless opt.nil?
  end

  def format_long_opt(opt)
    "`#{opt}`" unless opt.nil?
  end

  def format_usage_banner(usage_banner)
    usage_banner&.sub(/^(#: *\* )?/, "### ")
  end
end
