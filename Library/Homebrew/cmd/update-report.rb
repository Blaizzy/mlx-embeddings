# typed: false
# frozen_string_literal: true

require "formula_versions"
require "migrator"
require "formulary"
require "descriptions"
require "cleanup"
require "description_cache_store"
require "cli/parser"
require "settings"

module Homebrew
  extend T::Sig

  module_function

  def update_preinstall_header(args:)
    @update_preinstall_header ||= begin
      ohai_stdout_or_stderr "Auto-updated Homebrew!" if args.preinstall?
      true
    end
  end

  sig { returns(CLI::Parser) }
  def update_report_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        The Ruby implementation of `brew update`. Never called manually.
      EOS
      switch "--preinstall",
             description: "Run in 'auto-update' mode (faster, less output)."
      switch "-f", "--force",
             description: "Treat installed and updated formulae as if they are from "\
                          "the same taps and migrate them anyway."

      hide_from_man_page!
    end
  end

  def update_report
    args = update_report_args.parse

    if !Utils::Analytics.messages_displayed? &&
       !Utils::Analytics.disabled? &&
       !Utils::Analytics.no_message_output?

      ENV["HOMEBREW_NO_ANALYTICS_THIS_RUN"] = "1"
      # Use the shell's audible bell.
      print "\a"

      # Use an extra newline and bold to avoid this being missed.
      ohai_stdout_or_stderr "Homebrew has enabled anonymous aggregate formula and cask analytics."
      puts_stdout_or_stderr <<~EOS
        #{Tty.bold}Read the analytics documentation (and how to opt-out) here:
          #{Formatter.url("https://docs.brew.sh/Analytics")}#{Tty.reset}
        No analytics have been recorded yet (nor will be during this `brew` run).

      EOS

      # Consider the messages possibly missed if not a TTY.
      Utils::Analytics.messages_displayed! if $stdout.tty?
    end

    if Settings.read("donationmessage") != "true" && !args.quiet?
      ohai_stdout_or_stderr "Homebrew is run entirely by unpaid volunteers. Please consider donating:"
      puts_stdout_or_stderr "  #{Formatter.url("https://github.com/Homebrew/brew#donations")}\n"

      # Consider the message possibly missed if not a TTY.
      Settings.write "donationmessage", true if $stdout.tty?
    end

    install_core_tap_if_necessary

    updated = false
    new_repository_version = nil

    initial_revision = ENV["HOMEBREW_UPDATE_BEFORE"].to_s
    current_revision = ENV["HOMEBREW_UPDATE_AFTER"].to_s
    odie "update-report should not be called directly!" if initial_revision.empty? || current_revision.empty?

    if initial_revision != current_revision
      update_preinstall_header args: args
      puts_stdout_or_stderr \
        "Updated Homebrew from #{shorten_revision(initial_revision)} to #{shorten_revision(current_revision)}."
      updated = true

      old_tag = Settings.read "latesttag"

      new_tag = Utils.popen_read(
        "git", "-C", HOMEBREW_REPOSITORY, "tag", "--list", "--sort=-version:refname", "*.*"
      ).lines.first.chomp

      if new_tag != old_tag
        Settings.write "latesttag", new_tag
        new_repository_version = new_tag
      end
    end

    Homebrew.failed = true if ENV["HOMEBREW_UPDATE_FAILED"]
    return if ENV["HOMEBREW_DISABLE_LOAD_FORMULA"]

    hub = ReporterHub.new

    updated_taps = []
    Tap.each do |tap|
      next unless tap.git?

      begin
        reporter = Reporter.new(tap)
      rescue Reporter::ReporterRevisionUnsetError => e
        onoe "#{e.message}\n#{e.backtrace.join "\n"}" if Homebrew::EnvConfig.developer?
        next
      end
      if reporter.updated?
        updated_taps << tap.name
        hub.add(reporter, preinstall: args.preinstall?)
      end
    end

    unless updated_taps.empty?
      update_preinstall_header args: args
      puts_stdout_or_stderr \
        "Updated #{updated_taps.count} #{"tap".pluralize(updated_taps.count)} (#{updated_taps.to_sentence})."
      updated = true
    end

    if updated
      if hub.empty?
        puts_stdout_or_stderr "No changes to formulae." unless args.quiet?
      else
        hub.dump(updated_formula_report: !args.preinstall?)
        hub.reporters.each(&:migrate_tap_migration)
        hub.reporters.each { |r| r.migrate_formula_rename(force: args.force?, verbose: args.verbose?) }
        CacheStoreDatabase.use(:descriptions) do |db|
          DescriptionCacheStore.new(db)
                               .update_from_report!(hub)
        end

        unless args.preinstall?
          outdated_formulae = Formula.installed.count(&:outdated?)
          outdated_casks = Cask::Caskroom.casks.count(&:outdated?)
          update_pronoun = if (outdated_formulae + outdated_casks) == 1
            "it"
          else
            "them"
          end
          msg = ""
          if outdated_formulae.positive?
            msg += "#{Tty.bold}#{outdated_formulae}#{Tty.reset} outdated #{"formula".pluralize(outdated_formulae)}"
          end
          if outdated_casks.positive?
            msg += " and " if msg.present?
            msg += "#{Tty.bold}#{outdated_casks}#{Tty.reset} outdated #{"cask".pluralize(outdated_casks)}"
          end
          if msg.present?
            puts_stdout_or_stderr
            puts_stdout_or_stderr <<~EOS
              You have #{msg} installed.
              You can update #{update_pronoun} with #{Tty.bold}brew upgrade#{Tty.reset}.
            EOS
          end
        end
      end
      puts_stdout_or_stderr if args.preinstall?
    elsif !args.preinstall? && !ENV["HOMEBREW_UPDATE_FAILED"]
      puts_stdout_or_stderr "Already up-to-date." unless args.quiet?
    end

    Commands.rebuild_commands_completion_list
    link_completions_manpages_and_docs
    Tap.each(&:link_completions_and_manpages)

    failed_fetch_dirs = ENV["HOMEBREW_FAILED_FETCH_DIRS"]&.split("\n")
    if failed_fetch_dirs.present?
      failed_fetch_taps = failed_fetch_dirs.map { |dir| Tap.from_path(dir) }

      ofail <<~EOS
        Some taps failed to update!
        The following taps can not read their remote branches:
          #{failed_fetch_taps.join("\n  ")}
        This is happening because the remote branch was renamed or deleted.
        Reset taps to point to the correct remote branches by running `brew tap --repair`
      EOS
    end

    return if new_repository_version.blank?

    puts_stdout_or_stderr
    ohai_stdout_or_stderr "Homebrew was updated to version #{new_repository_version}"
    if new_repository_version.split(".").last == "0"
      puts_stdout_or_stderr <<~EOS
        More detailed release notes are available on the Homebrew Blog:
          #{Formatter.url("https://brew.sh/blog/#{new_repository_version}")}
      EOS
    else
      puts_stdout_or_stderr <<~EOS
        The changelog can be found at:
          #{Formatter.url("https://github.com/Homebrew/brew/releases/tag/#{new_repository_version}")}
      EOS
    end
  end

  def shorten_revision(revision)
    Utils.popen_read("git", "-C", HOMEBREW_REPOSITORY, "rev-parse", "--short", revision).chomp
  end

  def install_core_tap_if_necessary
    return if ENV["HOMEBREW_UPDATE_TEST"]

    core_tap = CoreTap.instance
    return if core_tap.installed?

    CoreTap.ensure_installed!
    revision = core_tap.git_head
    ENV["HOMEBREW_UPDATE_BEFORE_HOMEBREW_HOMEBREW_CORE"] = revision
    ENV["HOMEBREW_UPDATE_AFTER_HOMEBREW_HOMEBREW_CORE"] = revision
  end

  def link_completions_manpages_and_docs(repository = HOMEBREW_REPOSITORY)
    command = "brew update"
    Utils::Link.link_completions(repository, command)
    Utils::Link.link_manpages(repository, command)
    Utils::Link.link_docs(repository, command)
  rescue => e
    ofail <<~EOS
      Failed to link all completions, docs and manpages:
        #{e}
    EOS
  end
end

class Reporter
  class ReporterRevisionUnsetError < RuntimeError
    def initialize(var_name)
      super "#{var_name} is unset!"
    end
  end

  attr_reader :tap, :initial_revision, :current_revision

  def initialize(tap)
    @tap = tap

    initial_revision_var = "HOMEBREW_UPDATE_BEFORE#{tap.repo_var}"
    @initial_revision = ENV[initial_revision_var].to_s
    raise ReporterRevisionUnsetError, initial_revision_var if @initial_revision.empty?

    current_revision_var = "HOMEBREW_UPDATE_AFTER#{tap.repo_var}"
    @current_revision = ENV[current_revision_var].to_s
    raise ReporterRevisionUnsetError, current_revision_var if @current_revision.empty?
  end

  def report(preinstall: false)
    return @report if @report

    @report = Hash.new { |h, k| h[k] = [] }
    return @report unless updated?

    diff.each_line do |line|
      status, *paths = line.split
      src = Pathname.new paths.first
      dst = Pathname.new paths.last

      next unless dst.extname == ".rb"

      if paths.any? { |p| tap.cask_file?(p) }
        case status
        when "A"
          # Have a dedicated report array for new casks.
          @report[:AC] << tap.formula_file_to_name(src)
        when "D"
          # Have a dedicated report array for deleted casks.
          @report[:DC] << tap.formula_file_to_name(src)
        when "M"
          # Report updated casks
          @report[:MC] << tap.formula_file_to_name(src)
        end
      end

      next unless paths.any? { |p| tap.formula_file?(p) }

      case status
      when "A", "D"
        full_name = tap.formula_file_to_name(src)
        name = full_name.split("/").last
        new_tap = tap.tap_migrations[name]
        @report[status.to_sym] << full_name unless new_tap
      when "M"
        name = tap.formula_file_to_name(src)

        # Skip reporting updated formulae to speed up automatic updates.
        if preinstall
          @report[:M] << name
          next
        end

        begin
          formula = Formulary.factory(tap.path/src)
          new_version = formula.pkg_version
          old_version = FormulaVersions.new(formula).formula_at_revision(@initial_revision, &:pkg_version)
          next if new_version == old_version
        rescue FormulaUnavailableError
          # Don't care if the formula isn't available right now.
          nil
        rescue Exception => e # rubocop:disable Lint/RescueException
          onoe "#{e.message}\n#{e.backtrace.join "\n"}" if Homebrew::EnvConfig.developer?
        end

        @report[:M] << name
      when /^R\d{0,3}/
        src_full_name = tap.formula_file_to_name(src)
        dst_full_name = tap.formula_file_to_name(dst)
        # Don't report formulae that are moved within a tap but not renamed
        next if src_full_name == dst_full_name

        @report[:D] << src_full_name
        @report[:A] << dst_full_name
      end
    end

    renamed_formulae = Set.new
    @report[:D].each do |old_full_name|
      old_name = old_full_name.split("/").last
      new_name = tap.formula_renames[old_name]
      next unless new_name

      new_full_name = if tap.core_tap?
        new_name
      else
        "#{tap}/#{new_name}"
      end

      renamed_formulae << [old_full_name, new_full_name] if @report[:A].include? new_full_name
    end

    @report[:A].each do |new_full_name|
      new_name = new_full_name.split("/").last
      old_name = tap.formula_renames.key(new_name)
      next unless old_name

      old_full_name = if tap.core_tap?
        old_name
      else
        "#{tap}/#{old_name}"
      end

      renamed_formulae << [old_full_name, new_full_name]
    end

    unless renamed_formulae.empty?
      @report[:A] -= renamed_formulae.map(&:last)
      @report[:D] -= renamed_formulae.map(&:first)
      @report[:R] = renamed_formulae.to_a
    end

    @report
  end

  def updated?
    initial_revision != current_revision
  end

  def migrate_tap_migration
    (report[:D] + report[:DC]).each do |full_name|
      name = full_name.split("/").last
      new_tap_name = tap.tap_migrations[name]
      next if new_tap_name.nil? # skip if not in tap_migrations list.

      new_tap_user, new_tap_repo, new_tap_new_name = new_tap_name.split("/")
      new_name = if new_tap_new_name
        new_full_name = new_tap_new_name
        new_tap_name = "#{new_tap_user}/#{new_tap_repo}"
        new_tap_new_name
      else
        new_full_name = "#{new_tap_name}/#{name}"
        name
      end

      # This means it is a cask
      if report[:DC].include? full_name
        next unless (HOMEBREW_PREFIX/"Caskroom"/new_name).exist?

        new_tap = Tap.fetch(new_tap_name)
        new_tap.install unless new_tap.installed?
        ohai_stdout_or_stderr "#{name} has been moved to Homebrew.", <<~EOS
          To uninstall the cask, run:
            brew uninstall --cask --force #{name}
        EOS
        next if (HOMEBREW_CELLAR/new_name.split("/").last).directory?

        ohai_stdout_or_stderr "Installing #{new_name}..."
        system HOMEBREW_BREW_FILE, "install", new_full_name
        begin
          unless Formulary.factory(new_full_name).keg_only?
            system HOMEBREW_BREW_FILE, "link", new_full_name, "--overwrite"
          end
        rescue Exception => e # rubocop:disable Lint/RescueException
          onoe "#{e.message}\n#{e.backtrace.join "\n"}" if Homebrew::EnvConfig.developer?
        end
        next
      end

      next unless (dir = HOMEBREW_CELLAR/name).exist? # skip if formula is not installed.

      tabs = dir.subdirs.map { |d| Tab.for_keg(Keg.new(d)) }
      next unless tabs.first.tap == tap # skip if installed formula is not from this tap.

      new_tap = Tap.fetch(new_tap_name)
      # For formulae migrated to cask: Auto-install cask or provide install instructions.
      if new_tap_name.start_with?("homebrew/cask")
        if new_tap.installed? && (HOMEBREW_PREFIX/"Caskroom").directory?
          ohai_stdout_or_stderr "#{name} has been moved to Homebrew Cask."
          ohai_stdout_or_stderr "brew unlink #{name}"
          system HOMEBREW_BREW_FILE, "unlink", name
          ohai_stdout_or_stderr "brew cleanup"
          system HOMEBREW_BREW_FILE, "cleanup"
          ohai_stdout_or_stderr "brew install --cask #{new_name}"
          system HOMEBREW_BREW_FILE, "install", "--cask", new_name
          ohai <<~EOS
            #{name} has been moved to Homebrew Cask.
            The existing keg has been unlinked.
            Please uninstall the formula when convenient by running:
              brew uninstall --force #{name}
          EOS
        else
          ohai_stdout_or_stderr "#{name} has been moved to Homebrew Cask.", <<~EOS
            To uninstall the formula and install the cask, run:
              brew uninstall --force #{name}
              brew tap #{new_tap_name}
              brew install --cask #{new_name}
          EOS
        end
      else
        new_tap.install unless new_tap.installed?
        # update tap for each Tab
        tabs.each { |tab| tab.tap = new_tap }
        tabs.each(&:write)
      end
    end
  end

  def migrate_formula_rename(force:, verbose:)
    Formula.installed.each do |formula|
      next unless Migrator.needs_migration?(formula)

      oldname = formula.oldname
      oldname_rack = HOMEBREW_CELLAR/oldname

      if oldname_rack.subdirs.empty?
        oldname_rack.rmdir_if_possible
        next
      end

      new_name = tap.formula_renames[oldname]
      next unless new_name

      new_full_name = "#{tap}/#{new_name}"

      begin
        f = Formulary.factory(new_full_name)
      rescue Exception => e # rubocop:disable Lint/RescueException
        onoe "#{e.message}\n#{e.backtrace.join "\n"}" if Homebrew::EnvConfig.developer?
        next
      end

      Migrator.migrate_if_needed(f, force: force)
    end
  end

  private

  def diff
    Utils.popen_read(
      "git", "-C", tap.path, "diff-tree", "-r", "--name-status", "--diff-filter=AMDR",
      "-M85%", initial_revision, current_revision
    )
  end
end

class ReporterHub
  extend T::Sig

  extend Forwardable

  attr_reader :reporters

  sig { void }
  def initialize
    @hash = {}
    @reporters = []
  end

  def select_formula(key)
    @hash.fetch(key, [])
  end

  def add(reporter, preinstall: false)
    @reporters << reporter
    report = reporter.report(preinstall: preinstall).delete_if { |_k, v| v.empty? }
    @hash.update(report) { |_key, oldval, newval| oldval.concat(newval) }
  end

  delegate empty?: :@hash

  def dump(updated_formula_report: true)
    # Key Legend: Added (A), Copied (C), Deleted (D), Modified (M), Renamed (R)

    dump_formula_report :A, "New Formulae"
    if updated_formula_report
      dump_formula_report :M, "Updated Formulae"
    else
      updated = select_formula(:M).count
      if updated.positive?
        ohai_stdout_or_stderr "Updated Formulae",
                              "Updated #{updated} #{"formula".pluralize(updated)}."
      end
    end
    dump_formula_report :R, "Renamed Formulae"
    dump_formula_report :D, "Deleted Formulae"
    dump_formula_report :AC, "New Casks"
    if updated_formula_report
      dump_formula_report :MC, "Updated Casks"
    else
      updated = select_formula(:MC).count
      if updated.positive?
        ohai_stdout_or_stderr "Updated Casks",
                              "Updated #{updated} #{"cask".pluralize(updated)}."
      end
    end
    dump_formula_report :DC, "Deleted Casks"
  end

  private

  def dump_formula_report(key, title)
    only_installed = Homebrew::EnvConfig.update_report_only_installed?

    formulae = select_formula(key).sort.map do |name, new_name|
      # Format list items of renamed formulae
      case key
      when :R
        name = pretty_installed(name) if installed?(name)
        new_name = pretty_installed(new_name) if installed?(new_name)
        "#{name} -> #{new_name}" unless only_installed
      when :A
        name if !installed?(name) && !only_installed
      when :AC
        name.split("/").last if !cask_installed?(name) && !only_installed
      when :MC, :DC
        name = name.split("/").last
        if cask_installed?(name)
          pretty_installed(name)
        elsif !only_installed
          name
        end
      else
        if installed?(name)
          pretty_installed(name)
        elsif !only_installed
          name
        end
      end
    end.compact

    return if formulae.empty?

    # Dump formula list.
    ohai title, Formatter.columns(formulae.sort)
  end

  def installed?(formula)
    (HOMEBREW_CELLAR/formula.split("/").last).directory?
  end

  def cask_installed?(cask)
    (Cask::Caskroom.path/cask).directory?
  end
end
