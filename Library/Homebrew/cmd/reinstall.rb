#:  * `reinstall` [`--display-times`] <formula>:
#:    Uninstall and then install <formula> (with existing install options).
#:
#:    If `--display-times` is passed, install times for each formula are printed
#:    at the end of the run.

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
        `reinstall` [<option(s)>] <formula>:

        Uninstall and then install <formula> (with existing install options).
      EOS
      switch "-s", "--build-from-source",
        description: "Compile the formula> from source even if a bottle is available."
      switch "--display-times",
        description: "Print install times for each formula at the end of the run."
      switch :verbose
      switch :debug
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
