# frozen_string_literal: true

require "formula"
require "erb"
require "ostruct"
require "cli/parser"
# Require all commands
Dir.glob("#{HOMEBREW_LIBRARY_PATH}/{dev-,}cmd/*.rb").sort.each { |cmd| require cmd }

module Homebrew
  module_function

  SOURCE_PATH = (HOMEBREW_LIBRARY_PATH/"manpages").freeze
  TARGET_MAN_PATH = (HOMEBREW_REPOSITORY/"manpages").freeze
  TARGET_DOC_PATH = (HOMEBREW_REPOSITORY/"docs").freeze

  def man_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `man` [<options>]

        Generate Homebrew's manpages.
      EOS
      switch "--fail-if-changed",
             description: "Return a failing status code if changes are detected in the manpage outputs. This "\
                          "can be used to notify CI when the manpages are out of date. Additionally, "\
                          "the date used in new manpages will match those in the existing manpages (to allow "\
                          "comparison without factoring in the date)."
      switch "--link",
             description: "This is now done automatically by `brew update`."
      max_named 0
    end
  end

  def man
    man_args.parse

    odie "`brew man --link` is now done automatically by `brew update`." if args.link?

    regenerate_man_pages

    if system "git", "-C", HOMEBREW_REPOSITORY, "diff", "--quiet", "docs/Manpage.md", "manpages"
      puts "No changes to manpage output detected."
    elsif args.fail_if_changed?
      Homebrew.failed = true
    end
  end

  def regenerate_man_pages
    Homebrew.install_bundler_gems!

    markup = build_man_page
    convert_man_page(markup, TARGET_DOC_PATH/"Manpage.md")
    convert_man_page(markup, TARGET_MAN_PATH/"brew.1")

    cask_markup = (SOURCE_PATH/"brew-cask.1.md").read
    convert_man_page(cask_markup, TARGET_MAN_PATH/"brew-cask.1")
  end

  def build_man_page
    template = (SOURCE_PATH/"brew.1.md.erb").read
    variables = OpenStruct.new

    variables[:commands] = generate_cmd_manpages("#{HOMEBREW_LIBRARY_PATH}/cmd/*.{rb,sh}")
    variables[:developer_commands] = generate_cmd_manpages("#{HOMEBREW_LIBRARY_PATH}/dev-cmd/{*.rb,sh}")
    variables[:global_options] = global_options_manpage

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
    variables[:linux] =
      readme.read[%r{(Homebrew/brew's Linux maintainers .*\.)}, 1]
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

  def convert_man_page(markup, target)
    manual = target.basename(".1")
    organisation = "Homebrew"

    # Set the manpage date to the existing one if we're checking for changes.
    # This avoids the only change being e.g. a new date.
    date = if args.fail_if_changed? &&
              target.extname == ".1" && target.exist?
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
      if format_flag == "--markdown"
        ronn_output = ronn_output.gsub(%r{<var>(.*?)</var>}, "*`\\1`*")
                                 .gsub(/\n\n\n+/, "\n\n")
      elsif format_flag == "--roff"
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

  def generate_cmd_manpages(glob)
    cmd_paths = Pathname.glob(glob).sort
    man_page_lines = []
    man_args = Homebrew.args
    # preserve existing manpage order
    cmd_paths.sort_by(&method(:sort_key_for_path))
             .each do |cmd_path|
      cmd_args_method_name = cmd_arg_parser(cmd_path)

      cmd_man_page_lines = begin
        cmd_parser = Homebrew.send(cmd_args_method_name)
        next if cmd_parser.hide_from_man_page

        cmd_parser_manpage_lines(cmd_parser).join
      rescue NoMethodError => e
        raise if e.name != cmd_args_method_name

        nil
      end
      cmd_man_page_lines ||= cmd_comment_manpage_lines(cmd_path)

      man_page_lines << cmd_man_page_lines
    end
    Homebrew.args = man_args
    man_page_lines.compact.join("\n")
  end

  def cmd_arg_parser(cmd_path)
    "#{cmd_path.basename.to_s.gsub(".rb", "").tr("-", "_")}_args".to_sym
  end

  def cmd_parser_manpage_lines(cmd_parser)
    lines = [format_usage_banner(cmd_parser.usage_banner_text)]
    lines += cmd_parser.processed_options.map do |short, long, _, desc|
      next if !long.nil? && cmd_parser.global_option?(cmd_parser.option_to_name(long), desc)

      generate_option_doc(short, long, desc)
    end.reject(&:blank?)
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
      next if line.match?(/--(debug|force|help|quiet|verbose) /)

      # Format one option or a comma-separated pair of short and long options.
      lines << line.gsub(/^ +(-+[a-z-]+), (-+[a-z-]+) +/, "* `\\1`, `\\2`:\n  ")
                   .gsub(/^ +(-+[a-z-]+) +/, "* `\\1`:\n  ")
    end
    lines.last << "\n"
    lines
  end

  def global_options_manpage
    lines = ["These options are applicable across all sub-commands.\n"]
    lines += Homebrew::CLI::Parser.global_options.values.map do |names, _, desc|
      short, long = names
      generate_option_doc(short, long, desc)
    end
    lines.join("\n")
  end

  def generate_option_doc(short, long, desc)
    comma = (short && long) ? ", " : ""
    "* #{format_short_opt(short)}" + comma + "#{format_long_opt(long)}:" + "\n  " + desc + "\n"
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
