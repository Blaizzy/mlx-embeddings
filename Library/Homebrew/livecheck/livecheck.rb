# frozen_string_literal: true

require "livecheck/strategy"

module Homebrew
  # The `Livecheck` module consists of methods used by the `brew livecheck`
  # command. These methods receive print the requested livecheck information
  # for formulae.
  #
  # @api private
  module Livecheck
    module_function

    GITHUB_SPECIAL_CASES = %w[
      api.github.com
      /latest
      mednafen
      camlp5
      kotlin
      osrm-backend
      prometheus
      pyenv-virtualenv
      sysdig
      shairport-sync
      yuicompressor
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

    # Executes the livecheck logic for each formula in the `formulae_to_check` array
    # and prints the results.
    # @return [nil]
    def livecheck_formulae(formulae_to_check, args)
      # Identify any non-homebrew/core taps in use for current formulae
      non_core_taps = {}
      formulae_to_check.each do |f|
        next if f.tap.blank?
        next if f.tap.name == CoreTap.instance.name
        next if non_core_taps[f.tap.name]

        non_core_taps[f.tap.name] = f.tap
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
      formulae_checked = formulae_to_check.sort.map.with_index do |formula, i|
        if args.debug? && i.positive?
          puts <<~EOS

            ----------

          EOS
        end

        skip_result = skip_conditions(formula, args: args)
        next skip_result if skip_result != false

        formula.head.downloader.shutup! if formula.head?

        current = if formula.head?
          formula.any_installed_version.version.commit
        else
          formula.version
        end

        latest = if formula.stable?
          version_info = latest_version(formula, args: args)
          version_info[:latest] if version_info.present?
        else
          formula.head.downloader.fetch_last_commit
        end

        if latest.blank?
          no_versions_msg = "Unable to get versions"
          raise TypeError, no_versions_msg unless args.json?

          next version_info if version_info.is_a?(Hash) && version_info[:status] && version_info[:messages]

          next status_hash(formula, "error", [no_versions_msg], args: args)
        end

        if (m = latest.to_s.match(/(.*)-release$/)) && !current.to_s.match(/.*-release$/)
          latest = Version.new(m[1])
        end

        is_outdated = if formula.head?
          # A HEAD-only formula is considered outdated if the latest upstream
          # commit hash is different than the installed version's commit hash
          (current != latest)
        else
          (current < latest)
        end

        is_newer_than_upstream = formula.stable? && (current > latest)

        info = {
          formula: formula_name(formula, args: args),
          version: {
            current:             current.to_s,
            latest:              latest.to_s,
            outdated:            is_outdated,
            newer_than_upstream: is_newer_than_upstream,
          },
          meta:    {
            livecheckable: formula.livecheckable?,
          },
        }
        info[:meta][:head_only] = true if formula.head?
        info[:meta].merge!(version_info[:meta]) if version_info.present? && version_info.key?(:meta)

        next if args.newer_only? && !info[:version][:outdated]

        has_a_newer_upstream_version ||= true

        if args.json?
          info.except!(:meta) unless args.verbose?
          next info
        end

        print_latest_version(info, args: args)
        nil
      rescue => e
        Homebrew.failed = true

        if args.json?
          status_hash(formula, "error", [e.to_s], args: args)
        elsif !args.quiet?
          onoe "#{Tty.blue}#{formula_name(formula, args: args)}#{Tty.reset}: #{e}"
          nil
        end
      end

      if args.newer_only? && !has_a_newer_upstream_version && !args.debug? && !args.json?
        puts "No newer upstream versions."
      end

      puts JSON.generate(formulae_checked.compact) if args.json?
    end

    # Returns the fully-qualified name of a formula if the full_name argument is
    # provided, returns the name otherwise.
    # @return [String]
    def formula_name(formula, args:)
      args.full_name? ? formula.full_name : formula.name
    end

    def status_hash(formula, status_str, messages = nil, args:)
      status_hash = {
        formula: formula_name(formula, args: args),
        status:  status_str,
      }
      status_hash[:messages] = messages if messages.is_a?(Array)

      if args.verbose?
        status_hash[:meta] = {
          livecheckable: formula.livecheckable?,
        }
        status_hash[:meta][:head_only] = true if formula.head?
      end

      status_hash
    end

    # If a formula has to be skipped, it prints or returns a Hash contaning the reason
    # for doing so, else it returns false.
    # @return [Hash, nil, Boolean]
    def skip_conditions(formula, args:)
      if formula.deprecated? && !formula.livecheckable?
        return status_hash(formula, "deprecated", args: args) if args.json?

        puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : deprecated" unless args.quiet?
        return
      end

      if formula.versioned_formula? && !formula.livecheckable?
        return status_hash(formula, "versioned", args: args) if args.json?

        puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : versioned" unless args.quiet?
        return
      end

      if formula.head? && !formula.any_version_installed?
        head_only_msg = "HEAD only formula must be installed to be livecheckable"
        return status_hash(formula, "error", [head_only_msg], args: args) if args.json?

        puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : #{head_only_msg}" unless args.quiet?
        return
      end

      is_gist = formula.stable&.url&.include?("gist.github.com")
      if formula.livecheck.skip? || is_gist
        skip_msg = if formula.livecheck.skip_msg.is_a?(String) &&
                      formula.livecheck.skip_msg.present?
          formula.livecheck.skip_msg.to_s
        elsif is_gist
          "Stable URL is a GitHub Gist"
        else
          ""
        end

        return status_hash(formula, "skipped", (skip_msg.blank? ? nil : [skip_msg]), args: args) if args.json?

        unless args.quiet?
          puts "#{Tty.red}#{formula_name(formula, args: args)}#{Tty.reset} : skipped" \
              "#{" - #{skip_msg}" if skip_msg.present?}"
        end
        return
      end

      false
    end

    # Formats and prints the livecheck result for a formula.
    # @return [nil]
    def print_latest_version(info, args:)
      formula_s = "#{Tty.blue}#{info[:formula]}#{Tty.reset}"
      formula_s += " (guessed)" if !info[:meta][:livecheckable] && args.verbose?

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

      puts "#{formula_s} : #{current_s} ==> #{latest_s}"
    end

    # Returns an Array containing the formula URLs that can be used by livecheck.
    # @return [Array]
    def checkable_urls(formula)
      urls = []
      urls << formula.head.url if formula.head
      if formula.stable
        urls << formula.stable.url
        urls.concat(formula.stable.mirrors)
      end
      urls << formula.homepage if formula.homepage

      urls.compact
    end

    # Preprocesses and returns the URL used by livecheck.
    # @return [String]
    def preprocess_url(url)
      # Check for GitHub repos on github.com, not AWS
      url = url.sub("github.s3.amazonaws.com", "github.com") if url.include?("github")

      # Use repo from GitHub or GitLab inferred from download URL
      if url.include?("github.com") && GITHUB_SPECIAL_CASES.none? { |sc| url.include? sc }
        if url.include? "archive"
          url = url.sub(%r{/archive/.*}, ".git") if url.include? "github"
        elsif url.include? "releases"
          url = url.sub(%r{/releases/.*}, ".git")
        elsif url.include? "downloads"
          url = "#{Pathname.new(url.sub(%r{/downloads(.*)}, "\\1")).dirname}.git"
        elsif !url.end_with?(".git")
          # Truncate the URL at the user/repo part, if possible
          %r{(?<github_repo_url>(?:[a-z]+://)?github.com/[^/]+/[^/#]+)} =~ url
          url = github_repo_url if github_repo_url.present?

          url.delete_suffix!("/") if url.end_with?("/")
          url += ".git"
        end
      elsif url.include?("/-/archive/")
        url = url.sub(%r{/-/archive/.*$}i, ".git")
      end

      url
    end

    # Identifies the latest version of the formula and returns a Hash containing
    # the version information. Returns nil if a latest version couldn't be found.
    # @return [Hash, nil]
    def latest_version(formula, args:)
      has_livecheckable = formula.livecheckable?
      livecheck = formula.livecheck
      livecheck_regex = livecheck.regex
      livecheck_strategy = livecheck.strategy
      livecheck_url = livecheck.url

      urls = [livecheck_url] if livecheck_url.present?
      urls ||= checkable_urls(formula)

      if args.debug?
        puts
        puts "Formula:          #{formula_name(formula, args: args)}"
        puts "Head only?:       true" if formula.head?
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

        # Do not preprocess the URL when livecheck.strategy is set to :page_match
        url = if livecheck_strategy == :page_match
          original_url
        else
          preprocess_url(original_url)
        end

        strategies = Strategy.from_url(url, livecheck_regex.present?)
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

        if livecheck_strategy.present? && !strategies.include?(strategy)
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
