# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    # Helper functions for using the versions JSON API.
    #
    # @api private
    module Versions
      extend T::Sig

      module_function

      def formulae
        # The result is cached by Homebrew::API.fetch
        Homebrew::API.fetch "versions-formulae.json", json: true
      end

      def linux
        # The result is cached by Homebrew::API.fetch
        Homebrew::API.fetch "versions-linux.json", json: true
      end

      def casks
        # The result is cached by Homebrew::API.fetch
        Homebrew::API.fetch "versions-casks.json", json: true
      end

      sig { params(name: String).returns(T.nilable(PkgVersion)) }
      def latest_formula_version(name)
        versions = if OS.mac? || Homebrew::EnvConfig.force_homebrew_on_linux?
          formulae
        else
          linux
        end

        return unless versions.key? name

        version = Version.new(versions[name]["version"])
        revision = versions[name]["revision"]
        PkgVersion.new(version, revision)
      end

      sig { params(token: String).returns(T.nilable(Version)) }
      def latest_cask_version(token)
        return unless casks.key? token

        Version.new(casks[token]["version"])
      end
    end
  end
end
