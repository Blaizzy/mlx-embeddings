# typed: false
# frozen_string_literal: true

require "formula"
require "fetch"
require "cli/parser"
require "cask/download"

module Homebrew
  extend T::Sig

  extend Fetch

  module_function

  sig { returns(CLI::Parser) }
  def fetch_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Download a bottle (if available) or source packages for <formula>e
        and binaries for <cask>s. For files, also print SHA-256 checksums.
      EOS
      flag "--bottle-tag=",
           description: "Download a bottle for given tag."
      switch "--HEAD",
             description: "Fetch HEAD version instead of stable version."
      switch "-f", "--force",
             description: "Remove a previously cached version and re-fetch."
      switch "-v", "--verbose",
             description: "Do a verbose VCS checkout, if the URL represents a VCS. This is useful for " \
                          "seeing if an existing VCS cache has been updated."
      switch "--retry",
             description: "Retry if downloading fails or re-download if the checksum of a previously cached " \
                          "version no longer matches."
      switch "--deps",
             description: "Also download dependencies for any listed <formula>."
      switch "-s", "--build-from-source",
             description: "Download source packages rather than a bottle."
      switch "--build-bottle",
             description: "Download source packages (for eventual bottling) rather than a bottle."
      switch "--force-bottle",
             description: "Download a bottle if it exists for the current or newest version of macOS, " \
                          "even if it would not be used during installation."
      switch "--[no-]quarantine",
             description: "Disable/enable quarantining of downloads (default: enabled).",
             env:         :cask_opts_quarantine
      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."

      conflicts "--build-from-source", "--build-bottle", "--force-bottle", "--bottle-tag"
      conflicts "--cask", "--HEAD"
      conflicts "--cask", "--deps"
      conflicts "--cask", "-s"
      conflicts "--cask", "--build-bottle"
      conflicts "--cask", "--force-bottle"
      conflicts "--cask", "--bottle-tag"
      conflicts "--formula", "--cask"

      named_args [:formula, :cask], min: 1
    end
  end

  def fetch
    args = fetch_args.parse

    bucket = if args.deps?
      args.named.to_formulae_and_casks.flat_map do |formula_or_cask|
        case formula_or_cask
        when Formula
          f = formula_or_cask

          [f, *f.recursive_dependencies.map(&:to_formula)]
        else
          formula_or_cask
        end
      end
    else
      args.named.to_formulae_and_casks
    end.uniq

    puts "Fetching: #{bucket * ", "}" if bucket.size > 1
    bucket.each do |formula_or_cask|
      case formula_or_cask
      when Formula
        f = formula_or_cask

        f.print_tap_action verb: "Fetching"

        fetched_bottle = false
        if fetch_bottle?(f, args: args)
          begin
            f.clear_cache if args.force?
            f.fetch_bottle_tab
            fetch_formula(f.bottle_for_tag(args.bottle_tag&.to_sym), args: args)
          rescue Interrupt
            raise
          rescue => e
            raise if Homebrew::EnvConfig.developer?

            fetched_bottle = false
            onoe e.message
            opoo "Bottle fetch failed, fetching the source instead."
          else
            fetched_bottle = true
          end
        end

        next if fetched_bottle

        fetch_formula(f, args: args)

        f.resources.each do |r|
          fetch_resource(r, args: args)
          r.patches.each { |p| fetch_patch(p, args: args) if p.external? }
        end

        f.patchlist.each { |p| fetch_patch(p, args: args) if p.external? }
      else
        cask = formula_or_cask

        quarantine = args.quarantine?
        quarantine = true if quarantine.nil?

        download = Cask::Download.new(cask, quarantine: quarantine)
        fetch_cask(download, args: args)
      end
    end
  end

  def fetch_resource(r, args:)
    puts "Resource: #{r.name}"
    fetch_fetchable r, args: args
  rescue ChecksumMismatchError => e
    retry if retry_fetch?(r, args: args)
    opoo "Resource #{r.name} reports different sha256: #{e.expected}"
  end

  def fetch_formula(f, args:)
    fetch_fetchable f, args: args
  rescue ChecksumMismatchError => e
    retry if retry_fetch?(f, args: args)
    opoo "Formula reports different sha256: #{e.expected}"
  end

  def fetch_cask(cask_download, args:)
    fetch_fetchable cask_download, args: args
  rescue ChecksumMismatchError => e
    retry if retry_fetch?(cask_download, args: args)
    opoo "Cask reports different sha256: #{e.expected}"
  end

  def fetch_patch(p, args:)
    fetch_fetchable p, args: args
  rescue ChecksumMismatchError => e
    opoo "Patch reports different sha256: #{e.expected}"
    Homebrew.failed = true
  end

  def retry_fetch?(f, args:)
    @fetch_failed ||= Set.new
    if args.retry? && @fetch_failed.add?(f)
      ohai "Retrying download"
      f.clear_cache
      true
    else
      Homebrew.failed = true
      false
    end
  end

  def fetch_fetchable(f, args:)
    f.clear_cache if args.force?

    already_fetched = f.cached_download.exist?

    begin
      download = f.fetch(verify_download_integrity: false)
    rescue DownloadError
      retry if retry_fetch?(f, args: args)
      raise
    end

    return unless download.file?

    puts "Downloaded to: #{download}" unless already_fetched
    puts "SHA256: #{download.sha256}"

    f.verify_download_integrity(download)
  end
end
