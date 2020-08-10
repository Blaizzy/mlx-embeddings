# frozen_string_literal: true

module Cask
  class Cmd
    class Reinstall < Install
      def run
        self.class.reinstall_casks(
          *casks,
          binaries:       binaries?,
          verbose:        verbose?,
          force:          force?,
          skip_cask_deps: skip_cask_deps?,
          require_sha:    require_sha?,
          quarantine:     quarantine?,
        )
      end

      def self.reinstall_casks(
        *casks,
        verbose: false,
        force: false,
        skip_cask_deps: false,
        binaries: nil,
        require_sha: nil,
        quarantine: nil
      )
        # TODO: Handle this in `CLI::Parser`.
        binaries = Homebrew::EnvConfig.cask_opts_binaries? if binaries.nil?
        quarantine = Homebrew::EnvConfig.cask_opts_quarantine? if quarantine.nil?
        require_sha = Homebrew::EnvConfig.cask_opts_require_sha? if require_sha.nil?

        casks.each do |cask|
          Installer.new(cask,
                        binaries:       binaries,
                        verbose:        verbose,
                        force:          force,
                        skip_cask_deps: skip_cask_deps,
                        require_sha:    require_sha,
                        quarantine:     quarantine).reinstall
        end
      end

      def self.help
        "reinstalls the given Cask"
      end
    end
  end
end
