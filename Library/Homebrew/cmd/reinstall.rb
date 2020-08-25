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
  module_function

  def reinstall_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `reinstall` [<options>] <formula>

        Uninstall and then install <formula> using the same options it was originally
        installed with, plus any appended brew formula options.

        Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for the
        reinstalled formulae or, every 30 days, for all formulae.
      EOS
      switch "-d", "--debug",
             description: "If brewing fails, open an interactive debugging session with access to IRB "\
                          "or a shell inside the temporary build directory."
      switch "-s", "--build-from-source",
             description: "Compile <formula> from source even if a bottle is available."
      switch "-i", "--interactive",
             description: "Download and patch <formula>, then open a shell. This allows the user to "\
                          "run `./configure --help` and otherwise determine how to turn the software "\
                          "package into a Homebrew package."
      switch "--force-bottle",
             description: "Install from a bottle if it exists for the current or newest version of "\
                          "macOS, even if it would not normally be used for installation."
      switch "--keep-tmp",
             description: "Retain the temporary files created during installation."
      switch "-f", "--force",
             description: "Install without checking for previously installed keg-only or "\
                          "non-migrated versions."
      switch "-v", "--verbose",
             description: "Print the verification and postinstall steps."
      switch "--display-times",
             env:         :display_install_times,
             description: "Print install times for each formula at the end of the run."
      conflicts "--build-from-source", "--force-bottle"
      formula_options
      min_named :formula
    end
  end

  def reinstall
    args = reinstall_args.parse

    FormulaInstaller.prevent_build_flags(args)

    Install.perform_preinstall_checks

    resolved_formulae, casks = args.named.to_resolved_formulae_to_casks
    resolved_formulae.each do |f|
      if f.pinned?
        onoe "#{f.full_name} is pinned. You must unpin it to reinstall."
        next
      end
      Migrator.migrate_if_needed(f, force: args.force?)
      reinstall_formula(f, args: args)
      Cleanup.install_formula_clean!(f)
    end

    Upgrade.check_installed_dependents(args: args)

    Homebrew.messages.display_messages(display_times: args.display_times?)

    return if casks.blank?

    Cask::Cmd::Reinstall.reinstall_casks(
      *casks,
      binaries:       EnvConfig.cask_opts_binaries?,
      verbose:        args.verbose?,
      force:          args.force?,
      require_sha:    EnvConfig.cask_opts_require_sha?,
      skip_cask_deps: args.skip_cask_deps?,
      quarantine:     EnvConfig.cask_opts_quarantine?,
    )
  end
end
