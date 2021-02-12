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
      @problems  = []
    end

    def audit
      audit_version
      audit_download_strategy
      audit_checksum
      audit_urls
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

    def self.curl_openssl_and_deps
      @curl_openssl_and_deps ||= begin
        formulae_names = ["curl", "openssl"]
        formulae_names += formulae_names.flat_map do |f|
          Formula[f].recursive_dependencies.map(&:name)
        end
        formulae_names.uniq
      rescue FormulaUnavailableError
        []
      end
    end

    def audit_urls
      return unless @online

      urls = [url] + mirrors
      urls.each do |url|
        next if !@strict && mirrors.include?(url)

        strategy = DownloadStrategyDetector.detect(url, using)
        if strategy <= CurlDownloadStrategy && !url.start_with?("file")
          # A `brew mirror`'ed URL is usually not yet reachable at the time of
          # pull request.
          next if url.match?(%r{^https://dl.bintray.com/homebrew/mirror/})

          if (http_content_problem = curl_check_http_content(url, specs: specs))
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

    def problem(text)
      @problems << text
    end
  end
end
