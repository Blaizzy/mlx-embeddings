# typed: true
# frozen_string_literal: true

require "fileutils"
require "cask/cache"
require "cask/quarantine"

module Cask
  # A download corresponding to a {Cask}.
  #
  # @api private
  class Download
    include Context

    attr_reader :cask

    def initialize(cask, quarantine: nil)
      @cask = cask
      @quarantine = quarantine
    end

    def fetch(verify_download_integrity: true)
      downloaded_path = begin
        downloader.fetch
        downloader.cached_location
      rescue => e
        error = CaskError.new("Download failed on Cask '#{cask}' with message: #{e}")
        error.set_backtrace e.backtrace
        raise error
      end
      quarantine(downloaded_path)
      self.verify_download_integrity(downloaded_path) if verify_download_integrity
      downloaded_path
    end

    def downloader
      @downloader ||= begin
        strategy = DownloadStrategyDetector.detect(cask.url.to_s, cask.url.using)
        strategy.new(cask.url.to_s, cask.token, cask.version, cache: Cache.path, **cask.url.specs)
      end
    end

    def time_file_size
      downloader.resolved_time_file_size
    end

    def clear_cache
      downloader.clear_cache
    end

    def cached_download
      downloader.cached_location
    end

    def basename
      downloader.basename
    end

    def verify_download_integrity(fn)
      if @cask.sha256 == :no_check
        opoo "No checksum defined for cask '#{@cask}', skipping verification."
        return
      end

      begin
        ohai "Verifying checksum for cask '#{@cask}'" if verbose?
        fn.verify_checksum(@cask.sha256)
      rescue ChecksumMissingError
        opoo <<~EOS
          Cannot verify integrity of '#{fn.basename}'.
          No checksum was provided for this cask.
          For your reference, the checksum is:
            sha256 "#{fn.sha256}"
        EOS
      end
    end

    private

    def quarantine(path)
      return if @quarantine.nil?
      return unless Quarantine.available?

      if @quarantine
        Quarantine.cask!(cask: @cask, download_path: path)
      else
        Quarantine.release!(download_path: path)
      end
    end
  end
end
