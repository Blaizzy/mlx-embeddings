# typed: false
# frozen_string_literal: true

require "livecheck/strategy"
require "ruby-progressbar"
require "uri"

module Homebrew
  # The {Livecheck} module consists of methods used by the `brew livecheck`
  # command. These methods print the requested livecheck information
  # for formulae.
  #
  # @api private
  module Livecheck
    module_function

    GITEA_INSTANCES = %w[
      codeberg.org
      gitea.com
      opendev.org
      tildegit.org
    ].freeze

    GOGS_INSTANCES = %w[
      lolg.it
    ].freeze

    STRATEGY_SYMBOLS_TO_SKIP_PREPROCESS_URL = [
      :github_latest,
      :page_match,
    ].freeze

    UNSTABLE_VERSION_KEYWORDS = %w[
      alpha
      beta
      bpo
      dev
      experimental
      prerelease
      preview
      rc
    ].freeze

    # Executes the livecheck logic for each formula/cask in the
    # `formulae_and_casks_to_check` array and prints the results.
    # @return [nil]
    def livecheck_formulae_and_casks(formulae_and_casks_to_check, args)
      # Identify any non-homebrew/core taps in use for current formulae
      non_core_taps = {}
      formulae_and_casks_to_check.each do |fc|
        next if fc.tap.blank?
        next if fc.tap.name == CoreTap.instance.name
        next if non_core_taps[fc.tap.name]

        non_core_taps[fc.tap.name] = fc.tap
      end
      non_core_taps = non_core_taps.sort.to_h

      # Load additional Strategy files from taps
      non_core_taps.each_value do |tap|
        tap_strategy_path = "#{tap.path}/livecheck/strategy"
        Dir["#{tap_strategy_path}/*.rb"].sort.each(&method(:require)) if Dir.exist?(tap_strategy_path)
      end

      # Cache demodulized strategy names, to avoid repeating this work
      @livecheck_strategy_names = {}
      Strategy.constants.sort.each do |strategy_symbol|
        strategy = Strategy.const_get(strategy_symbol)
        @livecheck_strategy_names[strategy] = strategy.name.demodulize
      end
      @livecheck_strategy_names.freeze

      has_a_newer_upstream_version = false

      if args.json? && !args.quiet? && $stderr.tty?
        total_formulae = if formulae_and_casks_to_check == Formula
          formulae_and_casks_to_check.count
        else
          formulae_and_casks_to_check.length
        end

        Tty.with($stderr) do |stderr|
          stderr.puts Formatter.headline("Running checks", color: :blue)
        end

        progress = ProgressBar.create(
          total:          total_formulae,
          progress_mark:  "#",
          remainder_mark: ".",
          format:         " %t: [%B] %c/%C ",
          output:         $stderr,
        )
      end

      formulae_checked = formulae_and_casks_to_check.sort_by(&:name).map.with_index do |formula_or_cask, i|
        formula = formula_or_cask if formula_or_cask.is_a?(Formula)
        cask = formula_or_cask if formula_or_cask.is_a?(Cask::Cask)

        if args.debug? && i.positive?
          puts <<~EOS

            ----------

          EOS
        end

        skip_result = skip_conditions(formula_or_cask, args: args)
        next skip_result if skip_result != false

        formula.head&.downloader&.shutup!

        # Use the `stable` version for comparison except for installed
        # head-only formulae. A formula with `stable` and `head` that's
        # installed using `--head` will still use the `stable` version for
        # comparison.
        livecheck_version = formula_or_cask.livecheck.version
        current = if livecheck_version.is_a?(String)
          Version.new(livecheck_version)
        elsif livecheck_version.is_a?(Array)
          separator, method = livecheck_version
          Version.new(formula_or_cask.version.to_s.split(separator, 2).try(method))
        elsif formula
            if formula.head_only?
              formula.any_installed_version.version.commit
            else
              formula.stable.version
            end
        elsif livecheck_version.is_a?(Symbol)
          Version.new(Cask::DSL::Version.new(formula_or_cask.version).try(livecheck_version))
          else
            Version.new(formula_or_cask.version)
        end

        latest = if formula&.stable? || cask
          version_info = latest_version(formula_or_cask, args: args)
          version_info[:latest] if version_info.present?
        else
          formula.head.downloader.fetch_last_commit
        end

        if latest.blank?
          no_versions_msg = "Unable to get versions"
          raise TypeError, no_versions_msg unless args.json?

          next version_info if version_info.is_a?(Hash) && version_info[:status] && version_info[:messages]

          next status_hash(formula_or_cask, "error", [no_versions_msg], args: args)
        end

        if (m = latest.to_s.match(/(.*)-release$/)) && !current.to_s.match(/.*-release$/)
          latest = Version.new(m[1])
        end

        is_outdated = if formula&.head_only?
          # A HEAD-only formula is considered outdated if the latest upstream
          # commit hash is different than the installed version's commit hash
          (current != latest)
        else
          (current < latest)
        end

        is_newer_than_upstream = (formula&.stable? || cask) && (current > latest)

        info = {
          version: {
            current:             current.to_s,
            latest:              latest.to_s,
            outdated:            is_outdated,
            newer_than_upstream: is_newer_than_upstream,
          },
          meta:    {
            livecheckable: formula_or_cask.livecheckable?,
          },
        }
        info[:formula] = formula_name(formula, args: args) if formula
        info[:cask] = cask_name(cask, args: args) if cask
        info[:meta][:head_only] = true if formula&.head_only?
        info[:meta].merge!(version_info[:meta]) if version_info.present? && version_info.key?(:meta)

        next if args.newer_only? && !info[:version][:outdated]

        has_a_newer_upstream_version ||= true

        if args.json?
          progress&.increment
          info.except!(:meta) unless args.verbose?
          next info
        end

        print_latest_version(info, args: args)
        nil
      rescue => e
        Homebrew.failed = true

        if args.json?
          progress&.increment
          status_hash(formula_or_cask, "error", [e.to_s], args: args)
        elsif !args.quiet?
          onoe "#{Tty.blue}#{formula_or_cask_name(formula_or_cask, args: args)}#{Tty.reset}: #{e}"
          nil
        end
      end

      if args.newer_only? && !has_a_newer_upstream_version && !args.debug? && !args.json?
        puts "No newer upstream versions."
      end

      return unless args.json?

      if progress
        progress.finish
        Tty.with($stderr) do |stderr|
          stderr.print "#{Tty.up}#{Tty.erase_line}" * 2
        end
      end

      puts JSON.generate(formulae_checked.compact)
    end

    def formula_or_cask_name(formula_or_cask, args:)
      if formula_or_cask.is_a?(Formula)
        formula_name(formula_or_cask, args: args)
      else
        cask_name(formula_or_cask, args: args)
      end
    end

    def cask_name(cask, args:)
      args.full_name? ? cask.full_name : cask.token
    end

    # Returns the fully-qualified name of a formula if the `full_name` argument is
    # provided; returns the name otherwise.
    # @return [String]
    def formula_name(formula, args:)
      args.full_name? ? formula.full_name : formula.name
    end

    def status_hash(formula_or_cask, status_str, messages = nil, args:)
      formula = formula_or_cask if formula_or_cask.is_a?(Formula)

      status_hash = {
        status: status_str,
      }
      status_hash[:messages] = messages if messages.is_a?(Array)

      if formula
        status_hash[:formula] = formula_name(formula, args: args)
      else
        status_hash[:cask] = cask_name(formula_or_cask, args: args)
      end

      if args.verbose?
        status_hash[:meta] = {
          livecheckable: formula_or_cask.livecheckable?,
        }
        status_hash[:meta][:head_only] = true if formula&.head_only?
      end

      status_hash
    end

    # If a formula has to be skipped, it prints or returns a Hash contaning the reason
    # for doing so; returns false otherwise.
    # @return [Hash, nil, Boolean]
    def skip_conditions(formula_or_cask, args:)
      formula = formula_or_cask if formula_or_cask.is_a?(Formula)

      if formula&.deprecated? && !formula.livecheckable?
        return status_hash(formula, "deprecated", args: args) if args.json?

          puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : deprecated" unless args.quiet?
          return
        end

      if formula&.disabled? && !formula.livecheckable?
        return status_hash(formula, "disabled", args: args) if args.json?

        puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : disabled" unless args.quiet?
        return
      end

      if formula&.versioned_formula? && !formula.livecheckable?
        return status_hash(formula, "versioned", args: args) if args.json?

          puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : versioned" unless args.quiet?
          return
        end

      if formula&.head_only? && !formula.any_version_installed?
        head_only_msg = "HEAD only formula must be installed to be livecheckable"
        return status_hash(formula, "error", [head_only_msg], args: args) if args.json?

          puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : #{head_only_msg}" unless args.quiet?
          return
        end

      is_gist = formula&.stable&.url&.include?("gist.github.com")
      if formula_or_cask.livecheck.skip? || is_gist
        skip_msg = if formula_or_cask.livecheck.skip_msg.is_a?(String) &&
                      formula_or_cask.livecheck.skip_msg.present?
          formula_or_cask.livecheck.skip_msg.to_s
        elsif is_gist
          "Stable URL is a GitHub Gist"
        else
          ""
        end

        return status_hash(formula_or_cask, "skipped", (skip_msg.blank? ? nil : [skip_msg]), args: args) if args.json?

        unless args.quiet?
          puts "#{Tty.red}#{formula_or_cask_name(formula_or_cask, args: args)}#{Tty.reset} : skipped" \
              "#{" - #{skip_msg}" if skip_msg.present?}"
        end
          return
      end

      false
    end

    # Formats and prints the livecheck result for a formula.
    # @return [nil]
    def print_latest_version(info, args:)
      formula_or_cask_s = "#{Tty.blue}#{info[:formula] || info[:cask]}#{Tty.reset}"
      formula_or_cask_s += " (guessed)" if !info[:meta][:livecheckable] && args.verbose?

      current_s = if info[:version][:newer_than_upstream]
        "#{Tty.red}#{info[:version][:current]}#{Tty.reset}"
      else
        info[:version][:current]
      end

      latest_s = if info[:version][:outdated]
        "#{Tty.green}#{info[:version][:latest]}#{Tty.reset}"
      else
        info[:version][:latest]
      end

      puts "#{formula_or_cask_s} : #{current_s} ==> #{latest_s}"
    end

    # Returns an Array containing the formula URLs that can be used by livecheck.
    # @return [Array]
    def checkable_formula_urls(formula)
      urls = []
      urls << formula.head.url if formula.head
      if formula.stable
        urls << formula.stable.url
        urls.concat(formula.stable.mirrors)
      end
      urls << formula.homepage if formula.homepage

      urls.compact
    end

    def checkable_cask_urls(cask)
      urls = []
      urls << cask.appcast.to_s if cask.appcast
      urls << cask.url.to_s
      urls << cask.homepage if cask.homepage
      urls.compact
    end

    def checkable_urls(formula_or_cask)
      if formula_or_cask.is_a?(Formula)
        checkable_formula_urls(formula_or_cask)
      else
        checkable_cask_urls(formula_or_cask)
      end
    end

    # Preprocesses and returns the URL used by livecheck.
    # @return [String]
    def preprocess_url(url)
      begin
        uri = URI.parse url
      rescue URI::InvalidURIError
        return url
      end

      host = uri.host == "github.s3.amazonaws.com" ? "github.com" : uri.host
      path = uri.path.delete_prefix("/").delete_suffix(".git")
      scheme = uri.scheme

      if host.end_with?("github.com")
        return url if path.match? %r{/releases/latest/?$}

        owner, repo = path.delete_prefix("downloads/").split("/")
        url = "#{scheme}://#{host}/#{owner}/#{repo}.git"
      elsif host.end_with?(*GITEA_INSTANCES)
        return url if path.match? %r{/releases/latest/?$}

        owner, repo = path.split("/")
        url = "#{scheme}://#{host}/#{owner}/#{repo}.git"
      elsif host.end_with?(*GOGS_INSTANCES)
        owner, repo = path.split("/")
        url = "#{scheme}://#{host}/#{owner}/#{repo}.git"
      # sourcehut
      elsif host.end_with?("git.sr.ht")
        owner, repo = path.split("/")
        url = "#{scheme}://#{host}/#{owner}/#{repo}"
      # GitLab (gitlab.com or self-hosted)
      elsif path.include?("/-/archive/")
        url = url.sub(%r{/-/archive/.*$}i, ".git")
      end

      url
    end

    # Identifies the latest version of the formula and returns a Hash containing
    # the version information. Returns nil if a latest version couldn't be found.
    # @return [Hash, nil]
    def latest_version(formula_or_cask, args:)
      formula = formula_or_cask if formula_or_cask.is_a?(Formula)

      has_livecheckable = formula_or_cask.livecheckable?
      livecheck = formula_or_cask.livecheck
      livecheck_regex = livecheck.regex
      livecheck_strategy = livecheck.strategy
      livecheck_url = livecheck.url

      urls = [livecheck_url] if livecheck_url.present?
      urls ||= checkable_urls(formula_or_cask)

      if args.debug?
        puts
        if formula
          puts "Formula:          #{formula_name(formula, args: args)}"
          puts "Head only?:       true" if formula.head_only?
        else
          puts "Cask:             #{cask_name(formula_or_cask, args: args)}"
        end
        puts "Livecheckable?:   #{has_livecheckable ? "Yes" : "No"}"
      end

      urls.each_with_index do |original_url, i|
        if args.debug?
          puts
          puts "URL:              #{original_url}"
        end

        # Skip Gists until/unless we create a method of identifying revisions
        if original_url.include?("gist.github.com")
          odebug "Skipping: GitHub Gists are not supported"
          next
        end

        # Only preprocess the URL when it's appropriate
        url = if STRATEGY_SYMBOLS_TO_SKIP_PREPROCESS_URL.include?(livecheck_strategy)
          original_url
        else
          preprocess_url(original_url)
        end

        strategies = Strategy.from_url(
          url,
          livecheck_strategy: livecheck_strategy,
          regex_provided:     livecheck_regex.present?,
        )
        strategy = Strategy.from_symbol(livecheck_strategy)
        strategy ||= strategies.first
        strategy_name = @livecheck_strategy_names[strategy]

        if args.debug?
          puts "URL (processed):  #{url}" if url != original_url
          if strategies.present? && args.verbose?
            puts "Strategies:       #{strategies.map { |s| @livecheck_strategy_names[s] }.join(", ")}"
          end
          puts "Strategy:         #{strategy.blank? ? "None" : strategy_name}"
          puts "Regex:            #{livecheck_regex.inspect}" if livecheck_regex.present?
        end

        if livecheck_strategy == :page_match && livecheck_regex.blank?
          odebug "#{strategy_name} strategy requires a regex"
          next
        end

        if livecheck_strategy.present? && strategies.exclude?(strategy)
          odebug "#{strategy_name} strategy does not apply to this URL"
          next
        end

        next if strategy.blank?

        strategy_data = strategy.find_versions(url, livecheck_regex)
        match_version_map = strategy_data[:matches]
        regex = strategy_data[:regex]

        if strategy_data[:messages].is_a?(Array) && match_version_map.blank?
          puts strategy_data[:messages] unless args.json?
          next if i + 1 < urls.length

          return status_hash(formula, "error", strategy_data[:messages], args: args)
        end

        if args.debug?
          puts "URL (strategy):   #{strategy_data[:url]}" if strategy_data[:url] != url
          puts "Regex (strategy): #{strategy_data[:regex].inspect}" if strategy_data[:regex] != livecheck_regex
        end

        match_version_map.delete_if do |_match, version|
          next true if version.blank?
          next false if has_livecheckable

          UNSTABLE_VERSION_KEYWORDS.any? do |rejection|
            version.to_s.include?(rejection)
          end
        end

        if args.debug? && match_version_map.present?
          puts
          puts "Matched Versions:"

          if args.verbose?
            match_version_map.each do |match, version|
              puts "#{match} => #{version.inspect}"
            end
          else
            puts match_version_map.values.join(", ")
          end
        end

        next if match_version_map.blank?

        version_info = {
          latest: Version.new(match_version_map.values.max),
        }

        if args.json? && args.verbose?
          version_info[:meta] = {
            url:      {
              original: original_url,
            },
            strategy: strategy.blank? ? nil : strategy_name,
          }
          version_info[:meta][:url][:processed] = url if url != original_url
          version_info[:meta][:url][:strategy] = strategy_data[:url] if strategy_data[:url] != url
          if strategies.present?
            version_info[:meta][:strategies] = strategies.map { |s| @livecheck_strategy_names[s] }
          end
          version_info[:meta][:regex] = regex.inspect if regex.present?
        end

        return version_info
      end

      nil
    end
  end
end
