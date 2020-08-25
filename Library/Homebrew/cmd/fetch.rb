# frozen_string_literal: true

require "formula"
require "fetch"
require "cli/parser"

module Homebrew
  extend Fetch

  module_function

  def fetch_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `fetch` [<options>] <formula>

        Download a bottle (if available) or source packages for <formula>.
        For tarballs, also print SHA-256 checksums.
      EOS
      switch "--HEAD",
             description: "Fetch HEAD version instead of stable version."
      switch "--devel",
             description: "Fetch development version instead of stable version."
      switch "-f", "--force",
             description: "Remove a previously cached version and re-fetch."
      switch "-v", "--verbose",
             description: "Do a verbose VCS checkout, if the URL represents a VCS. This is useful for "\
                          "seeing if an existing VCS cache has been updated."
      switch "--retry",
             description: "Retry if downloading fails or re-download if the checksum of a previously cached "\
                          "version no longer matches."
      switch "--deps",
             description: "Also download dependencies for any listed <formula>."
      switch "-s", "--build-from-source",
             description: "Download source packages rather than a bottle."
      switch "--build-bottle",
             description: "Download source packages (for eventual bottling) rather than a bottle."
      switch "--force-bottle",
             description: "Download a bottle if it exists for the current or newest version of macOS, "\
                          "even if it would not be used during installation."

      conflicts "--devel", "--HEAD"
      conflicts "--build-from-source", "--build-bottle", "--force-bottle"
      min_named :formula
    end
  end

  def fetch
    args = fetch_args.parse

    if args.deps?
      bucket = []
      args.named.to_formulae.each do |f|
        bucket << f
        bucket.concat f.recursive_dependencies.map(&:to_formula)
      end
      bucket.uniq!
    else
      bucket = args.named.to_formulae
    end

    puts "Fetching: #{bucket * ", "}" if bucket.size > 1
    bucket.each do |f|
      f.print_tap_action verb: "Fetching"

      fetched_bottle = false
      if fetch_bottle?(f, args: args)
        begin
          fetch_formula(f.bottle, args: args)
        rescue Interrupt
          raise
        rescue => e
          raise if Homebrew::EnvConfig.developer?

          fetched_bottle = false
          onoe e.message
          opoo "Bottle fetch failed: fetching the source."
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
    end
  end

  def fetch_resource(r, args:)
    puts "Resource: #{r.name}"
    fetch_fetchable r, args: args
  rescue ChecksumMismatchError => e
    retry if retry_fetch?(r, args: args)
    opoo "Resource #{r.name} reports different #{e.hash_type}: #{e.expected}"
  end

  def fetch_formula(f, args:)
    fetch_fetchable f, args: args
  rescue ChecksumMismatchError => e
    retry if retry_fetch?(f, args: args)
    opoo "Formula reports different #{e.hash_type}: #{e.expected}"
  end

  def fetch_patch(p, args:)
    fetch_fetchable p, args: args
  rescue ChecksumMismatchError => e
    Homebrew.failed = true
    opoo "Patch reports different #{e.hash_type}: #{e.expected}"
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
    puts Checksum::TYPES.map { |t| "#{t.to_s.upcase}: #{download.send(t)}" }

    f.verify_download_integrity(download)
  end
end
