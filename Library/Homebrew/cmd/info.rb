# frozen_string_literal: true

require "missing_formula"
require "caveats"
require "cli/parser"
require "options"
require "formula"
require "keg"
require "tab"
require "json"

module Homebrew
  module_function

  VALID_DAYS = %w[30 90 365].freeze
  VALID_FORMULA_CATEGORIES = %w[install install-on-request build-error].freeze
  VALID_CATEGORIES = (VALID_FORMULA_CATEGORIES + %w[cask-install os-version]).freeze

  def info_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `info` [<options>] [<formula>]

        Display brief statistics for your Homebrew installation.

        If <formula> is provided, show summary of information about <formula>.
      EOS
      switch "--analytics",
             description: "List global Homebrew analytics data or, if specified, installation and "\
                          "build error data for <formula> (provided neither `HOMEBREW_NO_ANALYTICS` "\
                          "nor `HOMEBREW_NO_GITHUB_API` are set)."
      flag   "--days",
             depends_on:  "--analytics",
             description: "How many days of analytics data to retrieve. "\
                          "The value for <days> must be `30`, `90` or `365`. The default is `30`."
      flag   "--category",
             depends_on:  "--analytics",
             description: "Which type of analytics data to retrieve. "\
                          "The value for <category> must be `install`, `install-on-request` or `build-error`; "\
                          "`cask-install` or `os-version` may be specified if <formula> is not. "\
                          "The default is `install`."
      switch "--github",
             description: "Open the GitHub source page for <formula> in a browser. "\
                          "To view formula history locally: `brew log -p` <formula>"
      flag   "--json",
             description: "Print a JSON representation of <formula>. Currently the default and only accepted "\
                          "value for <version> is `v1`. See the docs for examples of using the JSON "\
                          "output: <https://docs.brew.sh/Querying-Brew>"
      switch "--installed",
             depends_on:  "--json",
             description: "Print JSON of formulae that are currently installed."
      switch "--all",
             depends_on:  "--json",
             description: "Print JSON of all available formulae."
      switch :verbose,
             description: "Show more verbose analytics data for <formula>."
      switch :debug
      conflicts "--installed", "--all"
    end
  end

  def info
    info_args.parse
    if args.days.present?
      raise UsageError, "days must be one of #{VALID_DAYS.join(", ")}" unless VALID_DAYS.include?(args.days)
    end

    if args.category.present?
      if ARGV.named.present? && !VALID_FORMULA_CATEGORIES.include?(args.category)
        raise UsageError, "category must be one of #{VALID_FORMULA_CATEGORIES.join(", ")} when querying formulae"
      end

      unless VALID_CATEGORIES.include?(args.category)
        raise UsageError, "category must be one of #{VALID_CATEGORIES.join(", ")}"
      end
    end

    if args.json
      raise UsageError, "invalid JSON version: #{args.json}" unless ["v1", true].include? args.json

      print_json
    elsif args.github?
      exec_browser(*ARGV.formulae.map { |f| github_info(f) })
    else
      print_info
    end
  end

  def print_info
    if ARGV.named.empty?
      if args.analytics?
        output_analytics
      elsif HOMEBREW_CELLAR.exist?
        count = Formula.racks.length
        puts "#{count} #{"keg".pluralize(count)}, #{HOMEBREW_CELLAR.dup.abv}"
      end
    else
      ARGV.named.each_with_index do |f, i|
        puts unless i.zero?
        begin
          formula = if f.include?("/") || File.exist?(f)
            Formulary.factory(f)
          else
            Formulary.find_with_priority(f)
          end
          if args.analytics?
            output_formula_analytics(formula)
          else
            info_formula(formula)
          end
        rescue FormulaUnavailableError => e
          if args.analytics?
            output_analytics(filter: f)
            next
          end
          ofail e.message
          # No formula with this name, try a missing formula lookup
          if (reason = MissingFormula.reason(f, show_info: true))
            $stderr.puts reason
          end
        end
      end
    end
  end

  def print_json
    ff = if args.all?
      Formula.sort
    elsif args.installed?
      Formula.installed.sort
    else
      ARGV.formulae
    end
    json = ff.map(&:to_hash)
    puts JSON.generate(json)
  end

  def github_remote_path(remote, path)
    if remote =~ %r{^(?:https?://|git(?:@|://))github\.com[:/](.+)/(.+?)(?:\.git)?$}
      "https://github.com/#{Regexp.last_match(1)}/#{Regexp.last_match(2)}/blob/master/#{path}"
    else
      "#{remote}/#{path}"
    end
  end

  def github_info(f)
    if f.tap
      if remote = f.tap.remote
        path = f.path.relative_path_from(f.tap.path)
        github_remote_path(remote, path)
      else
        f.path
      end
    else
      f.path
    end
  end

  def info_formula(f)
    specs = []

    if stable = f.stable
      s = "stable #{stable.version}"
      s += " (bottled)" if stable.bottled?
      specs << s
    end

    if devel = f.devel
      specs << "devel #{devel.version}"
    end

    specs << "HEAD" if f.head

    attrs = []
    attrs << "pinned at #{f.pinned_version}" if f.pinned?
    attrs << "keg-only" if f.keg_only?

    puts "#{f.full_name}: #{specs * ", "}#{" [#{attrs * ", "}]" unless attrs.empty?}"
    puts f.desc if f.desc
    puts Formatter.url(f.homepage) if f.homepage

    conflicts = f.conflicts.map do |c|
      reason = " (because #{c.reason})" if c.reason
      "#{c.name}#{reason}"
    end.sort!
    unless conflicts.empty?
      puts <<~EOS
        Conflicts with:
          #{conflicts.join("\n  ")}
      EOS
    end

    kegs = f.installed_kegs
    heads, versioned = kegs.partition { |k| k.version.head? }
    kegs = [
      *heads.sort_by { |k| -Tab.for_keg(k).time.to_i },
      *versioned.sort_by(&:version),
    ]
    if kegs.empty?
      puts "Not installed"
    else
      kegs.each do |keg|
        puts "#{keg} (#{keg.abv})#{" *" if keg.linked?}"
        tab = Tab.for_keg(keg).to_s
        puts "  #{tab}" unless tab.empty?
      end
    end

    puts "From: #{Formatter.url(github_info(f))}"

    unless f.deps.empty?
      ohai "Dependencies"
      %w[build required recommended optional].map do |type|
        deps = f.deps.send(type).uniq
        puts "#{type.capitalize}: #{decorate_dependencies deps}" unless deps.empty?
      end
    end

    unless f.requirements.to_a.empty?
      ohai "Requirements"
      %w[build required recommended optional].map do |type|
        reqs = f.requirements.select(&:"#{type}?")
        next if reqs.to_a.empty?

        puts "#{type.capitalize}: #{decorate_requirements(reqs)}"
      end
    end

    if !f.options.empty? || f.head || f.devel
      ohai "Options"
      Homebrew.dump_options_for_formula f
    end

    caveats = Caveats.new(f)
    ohai "Caveats", caveats.to_s unless caveats.empty?

    output_formula_analytics(f)
  end

  def formulae_api_json(endpoint)
    return if ENV["HOMEBREW_NO_ANALYTICS"] || ENV["HOMEBREW_NO_GITHUB_API"]

    output, = curl_output("--max-time", "5",
                          "https://formulae.brew.sh/api/#{endpoint}")
    return if output.blank?

    JSON.parse(output)
  rescue JSON::ParserError
    nil
  end

  def analytics_table(category, days, results, os_version: false, cask_install: false)
    oh1 "#{category} (#{days} days)"
    total_count = results.values.inject("+")
    formatted_total_count = format_count(total_count)
    formatted_total_percent = format_percent(100)

    index_header = "Index"
    count_header = "Count"
    percent_header = "Percent"
    name_with_options_header = if os_version
      "macOS Version"
    elsif cask_install
      "Token"
    else
      "Name (with options)"
    end

    total_index_footer = "Total"
    max_index_width = results.length.to_s.length
    index_width = [
      index_header.length,
      total_index_footer.length,
      max_index_width,
    ].max
    count_width = [
      count_header.length,
      formatted_total_count.length,
    ].max
    percent_width = [
      percent_header.length,
      formatted_total_percent.length,
    ].max
    name_with_options_width = Tty.width -
                              index_width -
                              count_width -
                              percent_width -
                              10 # spacing and lines

    formatted_index_header =
      format "%#{index_width}s", index_header
    formatted_name_with_options_header =
      format "%-#{name_with_options_width}s",
             name_with_options_header[0..name_with_options_width-1]
    formatted_count_header =
      format "%#{count_width}s", count_header
    formatted_percent_header =
      format "%#{percent_width}s", percent_header
    puts "#{formatted_index_header} | #{formatted_name_with_options_header} | "\
         "#{formatted_count_header} |  #{formatted_percent_header}"

    columns_line = "#{"-"*index_width}:|-#{"-"*name_with_options_width}-|-"\
                   "#{"-"*count_width}:|-#{"-"*percent_width}:"
    puts columns_line

    index = 0
    results.each do |name_with_options, count|
      index += 1
      formatted_index = format "%0#{max_index_width}d", index
      formatted_index = format "%-#{index_width}s", formatted_index
      formatted_name_with_options =
        format "%-#{name_with_options_width}s",
               name_with_options[0..name_with_options_width-1]
      formatted_count = format "%#{count_width}s", format_count(count)
      formatted_percent = if total_count.zero?
        format "%#{percent_width}s", format_percent(0)
      else
        format "%#{percent_width}s",
               format_percent((count.to_i * 100) / total_count.to_f)
      end
      puts "#{formatted_index} | #{formatted_name_with_options} | " \
           "#{formatted_count} | #{formatted_percent}%"
      next if index > 10
    end
    return unless results.length > 1

    formatted_total_footer =
      format "%-#{index_width}s", total_index_footer
    formatted_blank_footer =
      format "%-#{name_with_options_width}s", ""
    formatted_total_count_footer =
      format "%#{count_width}s", formatted_total_count
    formatted_total_percent_footer =
      format "%#{percent_width}s", formatted_total_percent
    puts "#{formatted_total_footer} | #{formatted_blank_footer} | "\
         "#{formatted_total_count_footer} | #{formatted_total_percent_footer}%"
  end

  def output_analytics(filter: nil)
    days = args.days || "30"
    category = args.category || "install"
    json = formulae_api_json("analytics/#{category}/#{days}d.json")
    return if json.blank? || json["items"].blank?

    os_version = category == "os-version"
    cask_install = category == "cask-install"
    results = {}
    json["items"].each do |item|
      key = if os_version
        item["os_version"]
      elsif cask_install
        item["cask"]
      else
        item["formula"]
      end
      if filter.present?
        next if key != filter && !key.start_with?("#{filter} ")
      end
      results[key] = item["count"].tr(",", "").to_i
    end

    if filter.present? && results.blank?
      onoe "No results matching `#{filter}` found!"
      return
    end

    analytics_table(category, days, results, os_version: os_version, cask_install: cask_install)
  end

  def output_formula_analytics(f)
    json = formulae_api_json("formula/#{f}.json")
    return if json.blank? || json["analytics"].blank?

    full_analytics = args.analytics? || args.verbose?

    ohai "Analytics"
    json["analytics"].each do |category, value|
      category = category.tr("_", "-")
      analytics = []

      value.each do |days, results|
        days = days.to_i
        if full_analytics
          if args.days.present?
            next if args.days&.to_i != days
          end
          if args.category.present?
            next if args.category != category
          end

          analytics_table(category, days, results)
        else
          total_count = results.values.inject("+")
          analytics << "#{number_readable(total_count)} (#{days} days)"
        end
      end

      puts "#{category}: #{analytics.join(", ")}" unless full_analytics
    end
  end

  def decorate_dependencies(dependencies)
    deps_status = dependencies.map do |dep|
      if dep.satisfied?([])
        pretty_installed(dep_display_s(dep))
      else
        pretty_uninstalled(dep_display_s(dep))
      end
    end
    deps_status.join(", ")
  end

  def decorate_requirements(requirements)
    req_status = requirements.map do |req|
      req_s = req.display_s
      req.satisfied? ? pretty_installed(req_s) : pretty_uninstalled(req_s)
    end
    req_status.join(", ")
  end

  def dep_display_s(dep)
    return dep.name if dep.option_tags.empty?

    "#{dep.name} #{dep.option_tags.map { |o| "--#{o}" }.join(" ")}"
  end

  def format_count(count)
    count.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
  end

  def format_percent(percent)
    format("%<percent>.2f", percent: percent)
  end
end
