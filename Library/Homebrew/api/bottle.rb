# typed: false
# frozen_string_literal: true

require "github_packages"

module Homebrew
  module API
    # Helper functions for using the bottle JSON API.
    #
    # @api private
    module Bottle
      class << self
        extend T::Sig

        sig { returns(String) }
        def bottle_api_path
          "bottle"
        end
        alias generic_bottle_api_path bottle_api_path

        GITHUB_PACKAGES_SHA256_REGEX = %r{#{GitHubPackages::URL_REGEX}.*/blobs/sha256:(?<sha256>\h{64})$}.freeze

        sig { params(name: String).returns(Hash) }
        def fetch(name)
          Homebrew::API.fetch "#{bottle_api_path}/#{name}.json"
        end

        sig { params(name: String).returns(T::Boolean) }
        def available?(name)
          fetch name
          true
        rescue ArgumentError
          false
        end

        sig { params(name: String).void }
        def fetch_bottles(name)
          hash = fetch(name)
          bottle_tag = Utils::Bottles.tag.to_s

          if !hash["bottles"].key?(bottle_tag) && !hash["bottles"].key?("all")
            odie "No bottle available for #{name} on the current OS"
          end

          download_bottle(hash, bottle_tag)

          hash["dependencies"].each do |dep_hash|
            existing_formula = begin
              Formulary.factory dep_hash["name"]
            rescue FormulaUnavailableError
              # The formula might not exist if it's not installed and homebrew/core isn't tapped
              nil
            end

            next if existing_formula.present? && existing_formula.latest_version_installed?

            download_bottle(dep_hash, bottle_tag)
          end
        end

        sig { params(url: String).returns(T.nilable(String)) }
        def checksum_from_url(url)
          match = url.match GITHUB_PACKAGES_SHA256_REGEX
          return if match.blank?

          match[:sha256]
        end

        sig { params(hash: Hash, tag: String).void }
        def download_bottle(hash, tag)
          bottle = hash["bottles"][tag]
          bottle ||= hash["bottles"]["all"]
          return if bottle.blank?

          sha256 = bottle["sha256"] || checksum_from_url(bottle["url"])
          bottle_filename = ::Bottle::Filename.new(hash["name"], hash["pkg_version"], tag, hash["rebuild"])

          resource = Resource.new hash["name"]
          resource.url bottle["url"]
          resource.sha256 sha256
          resource.version hash["pkg_version"]
          resource.downloader.resolved_basename = bottle_filename

          resource.fetch

          # Map the name of this formula to the local bottle path to allow the
          # formula to be loaded by passing just the name to `Formulary::factory`.
          Formulary.map_formula_name_to_local_bottle_path hash["name"], resource.downloader.cached_location
        end
      end
    end
  end
end

require "extend/os/api/bottle"
