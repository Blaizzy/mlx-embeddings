require "formula_installer"
require "development_tools"
require "messages"
require "reinstall"
require "cli_parser"
require "cleanup"

module Homebrew
  module_function

  def reinstall_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `reinstall` [<options>] <formula>

        Uninstall and then install <formula> (with existing and any appended install options).
      EOS
      switch :debug,
        description: "If brewing fails, open an interactive debugging session with access to IRB "\
                     "or a shell inside the temporary build directory"
      switch "-s", "--build-from-source",
        description: "Compile <formula> from source even if a bottle is available."
      switch "--force-bottle",
        description: "Install from a bottle if it exists for the current or newest version of "\
                     "macOS, even if it would not normally be used for installation."
      switch "--keep-tmp",
        description: "Don't delete the temporary files created during installation."
      switch :force,
        description: "Install without checking for previously installed keg-only or "\
                     "non-migrated versions."
      switch :verbose,
        description: "Print the verification and postinstall steps."
      switch "--display-times",
        description: "Print install times for each formula at the end of the run."
      conflicts "--build-from-source", "--force-bottle"
      formula_options
    end
  end

  def reinstall
    reinstall_args.parse

    FormulaInstaller.prevent_build_flags unless DevelopmentTools.installed?

    Install.perform_preinstall_checks

    ARGV.resolved_formulae.each do |f|
      if f.pinned?
        onoe "#{f.full_name} is pinned. You must unpin it to reinstall."
        next
      end
      Migrator.migrate_if_needed(f)
      reinstall_formula(f)
      Cleanup.install_formula_clean!(f)
    end
    Homebrew.messages.display_messages
  end
end
