# typed: true
# frozen_string_literal: true

module Homebrew
  # Auditor for checking common violations in {Resource}s.
  #
  # @api private
  class ResourceAuditor
    attr_reader :name, :version, :checksum, :url, :mirrors, :using, :specs, :owner, :spec_name, :problems

    def initialize(resource, spec_name, options = {})
      @name     = resource.name
      @version  = resource.version
      @checksum = resource.checksum
      @url      = resource.url
      @mirrors  = resource.mirrors
      @using    = resource.using
      @specs    = resource.specs
      @owner    = resource.owner
      @spec_name = spec_name
      @online    = options[:online]
      @strict    = options[:strict]
      @only      = options[:only]
      @except    = options[:except]
      @use_homebrew_curl = options[:use_homebrew_curl]
      @problems = []
    end

    def audit
      only_audits = @only
      except_audits = @except

      methods.map(&:to_s).grep(/^audit_/).each do |audit_method_name|
        name = audit_method_name.delete_prefix("audit_")
        next if only_audits&.exclude?(name)
        next if except_audits&.include?(name)

        send(audit_method_name)
      end

      self
    end

    def audit_version
      if version.nil?
        problem "missing version"
      elsif !version.detected_from_url?
        version_text = version
        version_url = Version.detect(url, **specs)
        if version_url.to_s == version_text.to_s && version.instance_of?(Version)
          problem "version #{version_text} is redundant with version scanned from URL"
        end
      end
    end

    def audit_download_strategy
      url_strategy = DownloadStrategyDetector.detect(url)

      if (using == :git || url_strategy == GitDownloadStrategy) && specs[:tag] && !specs[:revision]
        problem "Git should specify :revision when a :tag is specified."
      end

      return unless using

      if using == :cvs
        mod = specs[:module]

        problem "Redundant :module value in URL" if mod == name

        if url.match?(%r{:[^/]+$})
          mod = url.split(":").last

          if mod == name
            problem "Redundant CVS module appended to URL"
          else
            problem "Specify CVS module as `:module => \"#{mod}\"` instead of appending it to the URL"
          end
        end
      end

      return unless url_strategy == DownloadStrategyDetector.detect("", using)

      problem "Redundant :using value in URL"
    end

    def audit_checksum
      return if spec_name == :head
      return unless DownloadStrategyDetector.detect(url, using) <= CurlDownloadStrategy

      problem "Checksum is missing" if checksum.blank?
    end

    def self.curl_deps
      @curl_deps ||= begin
        ["curl"] + Formula["curl"].recursive_dependencies.map(&:name).uniq
      rescue FormulaUnavailableError
        []
      end
    end

    def audit_urls
      urls = [url] + mirrors

      curl_dep = self.class.curl_deps.include?(owner.name)
      # Ideally `ca-certificates` would not be excluded here, but sourcing a HTTP mirror was tricky.
      # Instead, we have logic elsewhere to pass `--insecure` to curl when downloading the certs.
      # TODO: try remove the OS/env conditional
      if Homebrew::SimulateSystem.simulating_or_running_on_macos? && spec_name == :stable &&
         owner.name != "ca-certificates" && curl_dep && !urls.find { |u| u.start_with?("http://") }
        problem "should always include at least one HTTP mirror"
      end

      return unless @online

      urls.each do |url|
        next if !@strict && mirrors.include?(url)

        strategy = DownloadStrategyDetector.detect(url, using)
        if strategy <= CurlDownloadStrategy && !url.start_with?("file")

          raise HomebrewCurlDownloadStrategyError, url if
            strategy <= HomebrewCurlDownloadStrategy && !Formula["curl"].any_version_installed?

          if (http_content_problem = curl_check_http_content(url,
                                                             "source URL",
                                                             specs:             specs,
                                                             use_homebrew_curl: @use_homebrew_curl))
            problem http_content_problem
          end
        elsif strategy <= GitDownloadStrategy
          problem "The URL #{url} is not a valid git URL" unless Utils::Git.remote_exists? url
        elsif strategy <= SubversionDownloadStrategy
          next unless DevelopmentTools.subversion_handles_most_https_certificates?
          next unless Utils::Svn.available?

          problem "The URL #{url} is not a valid svn URL" unless Utils::Svn.remote_exists? url
        end
      end
    end

    def audit_head_branch
      return unless @online
      return unless @strict
      return if spec_name != :head
      return unless Utils::Git.remote_exists?(url)
      return if specs[:tag].present?

      branch = Utils.popen_read("git", "ls-remote", "--symref", url, "HEAD")
                    .match(%r{ref: refs/heads/(.*?)\s+HEAD})[1]

      return if branch == specs[:branch]

      problem "Use `branch: \"#{branch}\"` to specify the default branch"
    end

    def problem(text)
      @problems << text
    end
  end
end
