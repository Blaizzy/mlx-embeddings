# typed: false
# frozen_string_literal: true

require "migrator"
require "formulary"
require "descriptions"
require "cleanup"
require "description_cache_store"
require "cli/parser"
require "settings"
require "linuxbrew-core-migration"

module Homebrew
  extend T::Sig

  module_function

  def auto_update_header(args:)
    @auto_update_header ||= begin
      ohai "Auto-updated Homebrew!" if args.auto_update?
      true
    end
  end

  sig { returns(CLI::Parser) }
  def update_report_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        The Ruby implementation of `brew update`. Never called manually.
      EOS
      switch "--auto-update", "--preinstall",
             description: "Run in 'auto-update' mode (faster, less output)."
      switch "-f", "--force",
             description: "Treat installed and updated formulae as if they are from " \
                          "the same taps and migrate them anyway."

      hide_from_man_page!
    end
  end

  def update_report
    return output_update_report if $stdout.tty?

    redirect_stdout($stderr) do
      output_update_report
    end
  end

  def output_update_report
    args = update_report_args.parse

    # Run `brew update` (again) if we've got a linuxbrew-core CoreTap
    if CoreTap.instance.installed? && CoreTap.instance.linuxbrew_core? &&
       ENV["HOMEBREW_LINUXBREW_CORE_MIGRATION"].blank?
      ohai "Re-running `brew update` for linuxbrew-core migration"

      if HOMEBREW_CORE_DEFAULT_GIT_REMOTE != Homebrew::EnvConfig.core_git_remote
        opoo <<~EOS
          HOMEBREW_CORE_GIT_REMOTE was set: #{Homebrew::EnvConfig.core_git_remote}.
          It has been unset for the migration.
          You may need to change this from a linuxbrew-core mirror to a homebrew-core one.

        EOS
      end
      ENV.delete("HOMEBREW_CORE_GIT_REMOTE")

      if HOMEBREW_BOTTLE_DEFAULT_DOMAIN != Homebrew::EnvConfig.bottle_domain
        opoo <<~EOS
          HOMEBREW_BOTTLE_DOMAIN was set: #{Homebrew::EnvConfig.bottle_domain}.
          It has been unset for the migration.
          You may need to change this from a Linuxbrew package mirror to a Homebrew one.

        EOS
      end
      ENV.delete("HOMEBREW_BOTTLE_DOMAIN")

      ENV["HOMEBREW_LINUXBREW_CORE_MIGRATION"] = "1"
      FileUtils.rm_f HOMEBREW_LOCKS/"update"

      update_args = []
      update_args << "--auto-update" if args.auto_update?
      update_args << "--force" if args.force?
      exec HOMEBREW_BREW_FILE, "update", *update_args
    end

    if !Utils::Analytics.messages_displayed? &&
       !Utils::Analytics.disabled? &&
       !Utils::Analytics.no_message_output?

      ENV["HOMEBREW_NO_ANALYTICS_THIS_RUN"] = "1"
      # Use the shell's audible bell.
      print "\a"

      # Use an extra newline and bold to avoid this being missed.
      ohai "Homebrew has enabled anonymous aggregate formula and cask analytics."
      puts <<~EOS
        #{Tty.bold}Read the analytics documentation (and how to opt-out) here:
          #{Formatter.url("https://docs.brew.sh/Analytics")}#{Tty.reset}
        No analytics have been recorded yet (nor will be during this `brew` run).

      EOS

      # Consider the messages possibly missed if not a TTY.
      Utils::Analytics.messages_displayed! if $stdout.tty?
    end

    if Settings.read("donationmessage") != "true" && !args.quiet?
      ohai "Homebrew is run entirely by unpaid volunteers. Please consider donating:"
      puts "  #{Formatter.url("https://github.com/Homebrew/brew#donations")}\n\n"

      # Consider the message possibly missed if not a TTY.
      Settings.write "donationmessage", true if $stdout.tty?
    end

    install_core_tap_if_necessary

    updated = false
    new_tag = nil

    initial_revision = ENV["HOMEBREW_UPDATE_BEFORE"].to_s
    current_revision = ENV["HOMEBREW_UPDATE_AFTER"].to_s
    odie "update-report should not be called directly!" if initial_revision.empty? || current_revision.empty?

    if initial_revision != current_revision
      auto_update_header args: args

      updated = true

      old_tag = Settings.read "latesttag"

      new_tag = Utils.popen_read(
        "git", "-C", HOMEBREW_REPOSITORY, "tag", "--list", "--sort=-version:refname", "*.*"
      ).lines.first.chomp

      Settings.write "latesttag", new_tag if new_tag != old_tag

      if new_tag == old_tag
        ohai "Updated Homebrew from #{shorten_revision(initial_revision)} to #{shorten_revision(current_revision)}."
      elsif old_tag.blank?
        ohai "Updated Homebrew from #{shorten_revision(initial_revision)} " \
             "to #{new_tag} (#{shorten_revision(current_revision)})."
      else
        ohai "Updated Homebrew from #{old_tag} (#{shorten_revision(initial_revision)}) " \
             "to #{new_tag} (#{shorten_revision(current_revision)})."
      end
    end

    Homebrew.failed = true if ENV["HOMEBREW_UPDATE_FAILED"]
    return if Homebrew::EnvConfig.disable_load_formula?

    migrate_gcc_dependents_if_needed

    hub = ReporterHub.new

    updated_taps = []
    Tap.each do |tap|
      next unless tap.git?
      next if (tap.core_tap? || tap == "homebrew/cask") && Homebrew::EnvConfig.install_from_api?

      if ENV["HOMEBREW_MIGRATE_LINUXBREW_FORMULAE"].present? && tap.core_tap? &&
         Settings.read("linuxbrewmigrated") != "true"
        ohai "Migrating formulae from linuxbrew-core to homebrew-core"

        LINUXBREW_CORE_MIGRATION_LIST.each do |name|
          begin
            formula = Formula[name]
          rescue FormulaUnavailableError
            next
          end
          next unless formula.any_version_installed?

          keg = formula.installed_kegs.last
          tab = Tab.for_keg(keg)
          # force a `brew upgrade` from the linuxbrew-core version to the homebrew-core version (even if lower)
          tab.source["versions"]["version_scheme"] = -1
          tab.write
        end

        Settings.write "linuxbrewmigrated", true
      end

      begin
        reporter = Reporter.new(tap)
      rescue Reporter::ReporterRevisionUnsetError => e
        onoe "#{e.message}\n#{e.backtrace.join "\n"}" if Homebrew::EnvConfig.developer?
        next
      end
      if reporter.updated?
        updated_taps << tap.name
        hub.add(reporter, auto_update: args.auto_update?)
      end
    end

    unless updated_taps.empty?
      auto_update_header args: args
      puts "Updated #{updated_taps.count} #{"tap".pluralize(updated_taps.count)} (#{updated_taps.to_sentence})."
      updated = true
    end

    if updated
      if hub.empty?
        puts "No changes to formulae." unless args.quiet?
      else
        if ENV.fetch("HOMEBREW_UPDATE_REPORT_ONLY_INSTALLED", false)
          opoo "HOMEBREW_UPDATE_REPORT_ONLY_INSTALLED is now the default behaviour, " \
               "so you can unset it from your environment."
        end

        hub.dump(updated_formula_report: !args.auto_update?) unless args.quiet?
        hub.reporters.each(&:migrate_tap_migration)
        hub.reporters.each { |r| r.migrate_formula_rename(force: args.force?, verbose: args.verbose?) }

        CacheStoreDatabase.use(:descriptions) do |db|
          DescriptionCacheStore.new(db)
                               .update_from_report!(hub)
        end
        CacheStoreDatabase.use(:cask_descriptions) do |db|
          CaskDescriptionCacheStore.new(db)
                                   .update_from_report!(hub)
        end
      end
      puts if args.auto_update?
    elsif !args.auto_update? && !ENV["HOMEBREW_UPDATE_FAILED"] && !ENV["HOMEBREW_MIGRATE_LINUXBREW_FORMULAE"]
      puts "Already up-to-date." unless args.quiet?
    end

    Commands.rebuild_commands_completion_list
    link_completions_manpages_and_docs
    Tap.each(&:link_completions_and_manpages)

    failed_fetch_dirs = ENV["HOMEBREW_MISSING_REMOTE_REF_DIRS"]&.split("\n")
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

    return if new_tag.blank? || new_tag == old_tag || args.quiet?

    puts

    new_major_version, new_minor_version, new_patch_version = new_tag.split(".").map(&:to_i)
    old_major_version, old_minor_version = (old_tag.split(".")[0, 2]).map(&:to_i) if old_tag.present?
    if old_tag.blank? || new_major_version > old_major_version \
        || new_minor_version > old_minor_version
      puts <<~EOS
        The #{new_major_version}.#{new_minor_version}.0 release notes are available on the Homebrew Blog:
          #{Formatter.url("https://brew.sh/blog/#{new_major_version}.#{new_minor_version}.0")}
      EOS
    end

    return if new_patch_version.zero?

    puts <<~EOS
      The #{new_tag} changelog can be found at:
        #{Formatter.url("https://github.com/Homebrew/brew/releases/tag/#{new_tag}")}
    EOS
  end

  def shorten_revision(revision)
    Utils.popen_read("git", "-C", HOMEBREW_REPOSITORY, "rev-parse", "--short", revision).chomp
  end

  def install_core_tap_if_necessary
    return if ENV["HOMEBREW_UPDATE_TEST"]
    return if Homebrew::EnvConfig.install_from_api?

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

  def migrate_gcc_dependents_if_needed
    return if OS.mac?
    return if Settings.read("gcc-rpaths.fixed") == "true"

    Formula.installed.each do |formula|
      next unless formula.tap&.core_tap?

      recursive_runtime_dependencies = Dependency.expand(
        formula,
        cache_key: "update-report",
      ) do |_, dependency|
        Dependency.prune if dependency.build? || dependency.test?
      end
      next unless recursive_runtime_dependencies.map(&:name).include? "gcc"

      keg = formula.installed_kegs.last
      tab = Tab.for_keg(keg)
      # Force reinstallation upon `brew upgrade` to fix the bottle RPATH.
      tab.source["versions"]["version_scheme"] = -1
      tab.write
    rescue TapFormulaUnavailableError
      nil
    end

    Settings.write "gcc-rpaths.fixed", true
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

  def report(auto_update: false)
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

    if renamed_formulae.present?
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
        ohai "#{name} has been moved to Homebrew.", <<~EOS
          To uninstall the cask, run:
            brew uninstall --cask --force #{name}
        EOS
        next if (HOMEBREW_CELLAR/new_name.split("/").last).directory?

        ohai "Installing #{new_name}..."
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
          ohai "#{name} has been moved to Homebrew Cask."
          ohai "brew unlink #{name}"
          system HOMEBREW_BREW_FILE, "unlink", name
          ohai "brew cleanup"
          system HOMEBREW_BREW_FILE, "cleanup"
          ohai "brew install --cask #{new_name}"
          system HOMEBREW_BREW_FILE, "install", "--cask", new_name
          ohai <<~EOS
            #{name} has been moved to Homebrew Cask.
            The existing keg has been unlinked.
            Please uninstall the formula when convenient by running:
              brew uninstall --force #{name}
          EOS
        else
          ohai "#{name} has been moved to Homebrew Cask.", <<~EOS
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

  def select_formula_or_cask(key)
    @hash.fetch(key, [])
  end

  def add(reporter, auto_update: false)
    @reporters << reporter
    report = reporter.report(auto_update: auto_update).delete_if { |_k, v| v.empty? }
    @hash.update(report) { |_key, oldval, newval| oldval.concat(newval) }
  end

  delegate empty?: :@hash

  def dump(updated_formula_report: true)
    report_all = Homebrew::EnvConfig.update_report_all_formulae?

    dump_new_formula_report
    dump_new_cask_report
    dump_renamed_formula_report if report_all
    dump_deleted_formula_report(report_all)
    dump_deleted_cask_report(report_all)

    outdated_formulae = nil
    outdated_casks = nil

    if updated_formula_report && report_all
      dump_modified_formula_report
      dump_modified_cask_report
    elsif updated_formula_report
      outdated_formulae = Formula.installed.select(&:outdated?).map(&:name)
      output_dump_formula_or_cask_report "Outdated Formulae", outdated_formulae

      outdated_casks = Cask::Caskroom.casks.select(&:outdated?).map(&:token)
      output_dump_formula_or_cask_report "Outdated Casks", outdated_casks
    elsif report_all
      if (changed_formulae = select_formula_or_cask(:M).count) && changed_formulae.positive?
        ohai "Modified Formulae", "Modified #{changed_formulae} #{"formula".pluralize(changed_formulae)}."
      end

      if (changed_casks = select_formula_or_cask(:MC).count) && changed_casks.positive?
        ohai "Modified Casks", "Modified #{changed_casks} #{"cask".pluralize(changed_casks)}."
      end
    else
      outdated_formulae = Formula.installed.select(&:outdated?).map(&:name)
      outdated_casks = Cask::Caskroom.casks.select(&:outdated?).map(&:token)
    end

    return if outdated_formulae.blank? && outdated_casks.blank?

    outdated_formulae = outdated_formulae.count
    outdated_casks = outdated_casks.count

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

    return if msg.blank?

    puts
    puts <<~EOS
      You have #{msg} installed.
      You can upgrade #{update_pronoun} with #{Tty.bold}brew upgrade#{Tty.reset}
      or list #{update_pronoun} with #{Tty.bold}brew outdated#{Tty.reset}.
    EOS
  end

  private

  def dump_new_formula_report
    formulae = select_formula_or_cask(:A).sort.reject { |name| installed?(name) }

    output_dump_formula_or_cask_report "New Formulae", formulae
  end

  def dump_new_cask_report
    casks = select_formula_or_cask(:AC).sort.map do |name|
      name.split("/").last unless cask_installed?(name)
    end.compact

    output_dump_formula_or_cask_report "New Casks", casks
  end

  def dump_renamed_formula_report
    formulae = select_formula_or_cask(:R).sort.map do |name, new_name|
      name = pretty_installed(name) if installed?(name)
      new_name = pretty_installed(new_name) if installed?(new_name)
      "#{name} -> #{new_name}"
    end

    output_dump_formula_or_cask_report "Renamed Formulae", formulae
  end

  def dump_deleted_formula_report(report_all)
    formulae = select_formula_or_cask(:D).sort.map do |name|
      if installed?(name)
        pretty_uninstalled(name)
      elsif report_all
        name
      end
    end.compact

    output_dump_formula_or_cask_report "Deleted Formulae", formulae
  end

  def dump_deleted_cask_report(report_all)
    casks = select_formula_or_cask(:DC).sort.map do |name|
      name = name.split("/").last
      if cask_installed?(name)
        pretty_uninstalled(name)
      elsif report_all
        name
      end
    end.compact

    output_dump_formula_or_cask_report "Deleted Casks", casks
  end

  def dump_modified_formula_report
    formulae = select_formula_or_cask(:M).sort.map do |name|
      if installed?(name)
        if outdated?(name)
          pretty_outdated(name)
        else
          pretty_installed(name)
        end
      else
        name
      end
    end

    output_dump_formula_or_cask_report "Modified Formulae", formulae
  end

  def dump_modified_cask_report
    casks = select_formula_or_cask(:MC).sort.map do |name|
      name = name.split("/").last
      if cask_installed?(name)
        if cask_outdated?(name)
          pretty_outdated(name)
        else
          pretty_installed(name)
        end
      else
        name
      end
    end

    output_dump_formula_or_cask_report "Modified Casks", casks
  end

  def output_dump_formula_or_cask_report(title, formulae_or_casks)
    return if formulae_or_casks.blank?

    ohai title, Formatter.columns(formulae_or_casks.sort)
  end

  def installed?(formula)
    (HOMEBREW_CELLAR/formula.split("/").last).directory?
  end

  def outdated?(formula)
    Formula[formula].outdated?
  rescue FormulaUnavailableError
    false
  end

  def cask_installed?(cask)
    (Cask::Caskroom.path/cask).directory?
  end

  def cask_outdated?(cask)
    Cask::CaskLoader.load(cask).outdated?
  rescue Cask::CaskError
    false
  end
end
