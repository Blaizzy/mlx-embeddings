# typed: false
# frozen_string_literal: true

require "cask_dependent"

module Cask
  class Cmd
    # Cask implementation of the `brew install` command.
    #
    # @api private
    class Install < AbstractCommand
      def self.install_casks(
        _casks,
        verbose: nil,
        force: nil,
        adopt: nil,
        binaries: nil,
        skip_cask_deps: nil,
        require_sha: nil,
        quarantine: nil,
        quiet: nil,
        zap: nil,
        dry_run: nil
      )
        odie "Installing casks is supported only on macOS"
      end
    end
  end
end
