# typed: false
# frozen_string_literal: true

require "formula_installer"
require "development_tools"
require "messages"
require "install"
require "reinstall"
require "cli/parser"
require "cleanup"
require "cask/cmd"
require "cask/utils"
require "cask/macos"
require "upgrade"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def reinstall_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Uninstall and then reinstall a <formula> or <cask> using the same options it was
        originally installed with, plus any appended options specific to a <formula>.

        Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for the
        reinstalled formulae or, every 30 days, for all formulae.
      EOS
      switch "-d", "--debug",
             description: "If brewing fails, open an interactive debugging session with access to IRB " \
                          "or a shell inside the temporary build directory."
      switch "-f", "--force",
             description: "Install without checking for previously installed keg-only or " \
                          "non-migrated versions."
      switch "-v", "--verbose",
             description: "Print the verification and postinstall steps."
      [
        [:switch, "--formula", "--formulae", { description: "Treat all named arguments as formulae." }],
        [:switch, "-s", "--build-from-source", {
          description: "Compile <formula> from source even if a bottle is available.",
        }],
        [:switch, "-i", "--interactive", {
          description: "Download and patch <formula>, then open a shell. This allows the user to " \
                       "run `./configure --help` and otherwise determine how to turn the software " \
                       "package into a Homebrew package.",
        }],
        [:switch, "--force-bottle", {
          description: "Install from a bottle if it exists for the current or newest version of " \
                       "macOS, even if it would not normally be used for installation.",
        }],
        [:switch, "--keep-tmp", {
          description: "Retain the temporary files created during installation.",
        }],
        [:switch, "--display-times", {
          env:         :display_install_times,
          description: "Print install times for each formula at the end of the run.",
        }],
      ].each do |options|
        send(*options)
        conflicts "--cask", options[-2]
      end
      formula_options
      [
        [:switch, "--cask", "--casks", { description: "Treat all named arguments as casks." }],
        *Cask::Cmd::AbstractCommand::OPTIONS,
        *Cask::Cmd::Install::OPTIONS,
      ].each do |options|
        send(*options)
        conflicts "--formula", options[-2]
      end
      cask_options

      conflicts "--build-from-source", "--force-bottle"

      named_args [:formula, :cask], min: 1
    end
  end

  def reinstall
    args = reinstall_args.parse

    FormulaInstaller.prevent_build_flags(args)

    Install.perform_preinstall_checks

    formulae, casks = args.named.to_formulae_and_casks(method: :resolve)
                          .partition { |o| o.is_a?(Formula) }

    formulae.each do |f|
      if f.pinned?
        onoe "#{f.full_name} is pinned. You must unpin it to reinstall."
        next
      end
      Migrator.migrate_if_needed(f, force: args.force?)
      reinstall_formula(f, args: args)
      Cleanup.install_formula_clean!(f)
    end

    Upgrade.check_installed_dependents(formulae, args: args)

    if casks.any?
      Cask::Cmd::Reinstall.reinstall_casks(
        *casks,
        binaries:       args.binaries?,
        verbose:        args.verbose?,
        force:          args.force?,
        require_sha:    args.require_sha?,
        skip_cask_deps: args.skip_cask_deps?,
        quarantine:     args.quarantine?,
      )
    end

    Homebrew.messages.display_messages(display_times: args.display_times?)
  end
end
