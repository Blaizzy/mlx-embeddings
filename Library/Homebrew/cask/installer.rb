# typed: false
# frozen_string_literal: true

require "formula_installer"
require "unpack_strategy"

require "cask/topological_hash"
require "cask/config"
require "cask/download"
require "cask/staged"
require "cask/quarantine"

require "cgi"

module Cask
  # Installer for a {Cask}.
  #
  # @api private
  class Installer
    extend T::Sig

    extend Predicable
    # TODO: it is unwise for Cask::Staged to be a module, when we are
    #       dealing with both staged and unstaged casks here. This should
    #       either be a class which is only sometimes instantiated, or there
    #       should be explicit checks on whether staged state is valid in
    #       every method.
    include Staged

    def initialize(cask, command: SystemCommand, force: false,
                   skip_cask_deps: false, binaries: true, verbose: false,
                   require_sha: false, upgrade: false,
                   installed_as_dependency: false, quarantine: true,
                   verify_download_integrity: true)
      @cask = cask
      @command = command
      @force = force
      @skip_cask_deps = skip_cask_deps
      @binaries = binaries
      @verbose = verbose
      @require_sha = require_sha
      @reinstall = false
      @upgrade = upgrade
      @installed_as_dependency = installed_as_dependency
      @quarantine = quarantine
      @verify_download_integrity = verify_download_integrity
    end

    attr_predicate :binaries?, :force?, :skip_cask_deps?, :require_sha?,
                   :reinstall?, :upgrade?, :verbose?, :installed_as_dependency?,
                   :quarantine?

    def self.caveats(cask)
      odebug "Printing caveats"

      caveats = cask.caveats
      return if caveats.empty?

      <<~EOS
        #{ohai_title "Caveats"}
        #{caveats}
      EOS
    end

    def fetch
      odebug "Cask::Installer#fetch"

      verify_has_sha if require_sha? && !force?
      satisfy_dependencies

      download
    end

    def stage
      odebug "Cask::Installer#stage"

      Caskroom.ensure_caskroom_exists

      extract_primary_container
      save_caskfile
    rescue => e
      purge_versioned_files
      raise e
    end

    def install
      odebug "Cask::Installer#install"

      old_config = @cask.config

      raise CaskAlreadyInstalledError, @cask if @cask.installed? && !force? && !reinstall? && !upgrade?

      check_conflicts

      print caveats
      fetch
      uninstall_existing_cask if reinstall?

      backup if force? && @cask.staged_path.exist? && @cask.metadata_versioned_path.exist?

      oh1 "Installing Cask #{Formatter.identifier(@cask)}"
      opoo "macOS's Gatekeeper has been disabled for this Cask" unless quarantine?
      stage

      @cask.config = @cask.default_config.merge(old_config)

      install_artifacts

      ::Utils::Analytics.report_event("cask_install", @cask.token) unless @cask.tap&.private?

      purge_backed_up_versioned_files

      puts summary
    rescue
      restore_backup
      raise
    end

    def check_conflicts
      return unless @cask.conflicts_with

      @cask.conflicts_with[:cask].each do |conflicting_cask|
        if (match = conflicting_cask.match(HOMEBREW_TAP_CASK_REGEX))
          conflicting_cask_tap = Tap.fetch(match[1], match[2])
          next unless conflicting_cask_tap.installed?
        end

        conflicting_cask = CaskLoader.load(conflicting_cask)
        raise CaskConflictError.new(@cask, conflicting_cask) if conflicting_cask.installed?
      rescue CaskUnavailableError
        next # Ignore conflicting Casks that do not exist.
      end
    end

    def reinstall
      odebug "Cask::Installer#reinstall"
      @reinstall = true
      install
    end

    def uninstall_existing_cask
      return unless @cask.installed?

      # use the same cask file that was used for installation, if possible
      installed_caskfile = @cask.installed_caskfile
      installed_cask = installed_caskfile.exist? ? CaskLoader.load(installed_caskfile) : @cask

      # Always force uninstallation, ignore method parameter
      Installer.new(installed_cask, binaries: binaries?, verbose: verbose?, force: true, upgrade: upgrade?).uninstall
    end

    sig { returns(String) }
    def summary
      s = +""
      s << "#{Homebrew::EnvConfig.install_badge}  " unless Homebrew::EnvConfig.no_emoji?
      s << "#{@cask} was successfully #{upgrade? ? "upgraded" : "installed"}!"
      s.freeze
    end

    sig { returns(Download) }
    def downloader
      @downloader ||= Download.new(@cask, quarantine: quarantine?)
    end

    sig { returns(Pathname) }
    def download
      @download ||= downloader.fetch(verify_download_integrity: @verify_download_integrity)
    end

    def verify_has_sha
      odebug "Checking cask has checksum"
      return unless @cask.sha256 == :no_check

      raise CaskError, <<~EOS
        Cask '#{@cask}' does not have a sha256 checksum defined and was not installed.
        This means you have the #{Formatter.identifier("--require-sha")} option set, perhaps in your HOMEBREW_CASK_OPTS.
      EOS
    end

    def primary_container
      @primary_container ||= begin
        downloaded_path = download
        UnpackStrategy.detect(downloaded_path, type: @cask.container&.type, merge_xattrs: true)
      end
    end

    def extract_primary_container(to: @cask.staged_path)
      odebug "Extracting primary container"

      odebug "Using container class #{primary_container.class} for #{primary_container.path}"

      basename = downloader.basename

      if nested_container = @cask.container&.nested
        Dir.mktmpdir do |tmpdir|
          tmpdir = Pathname(tmpdir)
          primary_container.extract(to: tmpdir, basename: basename, verbose: verbose?)

          FileUtils.chmod_R "+rw", tmpdir/nested_container, force: true, verbose: verbose?

          UnpackStrategy.detect(tmpdir/nested_container, merge_xattrs: true)
                        .extract_nestedly(to: to, verbose: verbose?)
        end
      else
        primary_container.extract_nestedly(to: to, basename: basename, verbose: verbose?)
      end

      return unless quarantine?
      return unless Quarantine.available?

      Quarantine.propagate(from: primary_container.path, to: to)
    end

    def install_artifacts
      artifacts = @cask.artifacts
      already_installed_artifacts = []

      odebug "Installing artifacts"
      odebug "#{artifacts.length} #{"artifact".pluralize(artifacts.length)} defined", artifacts

      artifacts.each do |artifact|
        next unless artifact.respond_to?(:install_phase)

        odebug "Installing artifact of class #{artifact.class}"

        next if artifact.is_a?(Artifact::Binary) && !binaries?

        artifact.install_phase(command: @command, verbose: verbose?, force: force?)
        already_installed_artifacts.unshift(artifact)
      end

      save_config_file
    rescue => e
      begin
        already_installed_artifacts.each do |artifact|
          if artifact.respond_to?(:uninstall_phase)
            odebug "Reverting installation of artifact of class #{artifact.class}"
            artifact.uninstall_phase(command: @command, verbose: verbose?, force: force?)
          end

          next unless artifact.respond_to?(:post_uninstall_phase)

          odebug "Reverting installation of artifact of class #{artifact.class}"
          artifact.post_uninstall_phase(command: @command, verbose: verbose?, force: force?)
        end
      ensure
        purge_versioned_files
        raise e
      end
    end

    # TODO: move dependencies to a separate class,
    #       dependencies should also apply for `brew cask stage`,
    #       override dependencies with `--force` or perhaps `--force-deps`
    def satisfy_dependencies
      return unless @cask.depends_on

      macos_dependencies
      arch_dependencies
      x11_dependencies
      cask_and_formula_dependencies
    end

    def macos_dependencies
      return unless @cask.depends_on.macos
      return if @cask.depends_on.macos.satisfied?

      raise CaskError, @cask.depends_on.macos.message(type: :cask)
    end

    def arch_dependencies
      return if @cask.depends_on.arch.nil?

      @current_arch ||= { type: Hardware::CPU.type, bits: Hardware::CPU.bits }
      return if @cask.depends_on.arch.any? do |arch|
        arch[:type] == @current_arch[:type] &&
        Array(arch[:bits]).include?(@current_arch[:bits])
      end

      raise CaskError,
            "Cask #{@cask} depends on hardware architecture being one of " \
            "[#{@cask.depends_on.arch.map(&:to_s).join(", ")}], " \
            "but you are running #{@current_arch}."
    end

    def x11_dependencies
      return unless @cask.depends_on.x11
      raise CaskX11DependencyError, @cask.token unless MacOS::XQuartz.installed?
    end

    def graph_dependencies(cask_or_formula, acc = TopologicalHash.new)
      return acc if acc.key?(cask_or_formula)

      if cask_or_formula.is_a?(Cask)
        formula_deps = cask_or_formula.depends_on.formula.map { |f| Formula[f] }
        cask_deps = cask_or_formula.depends_on.cask.map { |c| CaskLoader.load(c, config: nil) }
      else
        formula_deps = cask_or_formula.deps.reject(&:build?).map(&:to_formula)
        cask_deps = cask_or_formula.requirements.map(&:cask).compact
                                   .map { |c| CaskLoader.load(c, config: nil) }
      end

      acc[cask_or_formula] ||= []
      acc[cask_or_formula] += formula_deps
      acc[cask_or_formula] += cask_deps

      formula_deps.each do |f|
        graph_dependencies(f, acc)
      end

      cask_deps.each do |c|
        graph_dependencies(c, acc)
      end

      acc
    end

    def collect_cask_and_formula_dependencies
      return @cask_and_formula_dependencies if @cask_and_formula_dependencies

      graph = graph_dependencies(@cask)

      raise CaskSelfReferencingDependencyError, cask.token if graph[@cask].include?(@cask)

      primary_container.dependencies.each do |dep|
        graph_dependencies(dep, graph)
      end

      begin
        @cask_and_formula_dependencies = graph.tsort - [@cask]
      rescue TSort::Cyclic
        strongly_connected_components = graph.strongly_connected_components.sort_by(&:count)
        cyclic_dependencies = strongly_connected_components.last - [@cask]
        raise CaskCyclicDependencyError.new(@cask.token, cyclic_dependencies.to_sentence)
      end
    end

    def missing_cask_and_formula_dependencies
      collect_cask_and_formula_dependencies.reject do |cask_or_formula|
        installed = if cask_or_formula.respond_to?(:any_version_installed?)
          cask_or_formula.any_version_installed?
        else
          cask_or_formula.try(:installed?)
        end
        installed && (cask_or_formula.respond_to?(:optlinked?) ? cask_or_formula.optlinked? : true)
      end
    end

    def cask_and_formula_dependencies
      return if installed_as_dependency?

      formulae_and_casks = collect_cask_and_formula_dependencies

      return if formulae_and_casks.empty?

      missing_formulae_and_casks = missing_cask_and_formula_dependencies

      if missing_formulae_and_casks.empty?
        puts "All formula dependencies satisfied."
        return
      end

      ohai "Installing dependencies: #{missing_formulae_and_casks.map(&:to_s).join(", ")}"
      missing_formulae_and_casks.each do |cask_or_formula|
        if cask_or_formula.is_a?(Cask)
          if skip_cask_deps?
            opoo "`--skip-cask-deps` is set; skipping installation of #{@cask}."
            next
          end

          Installer.new(
            cask_or_formula,
            binaries:                binaries?,
            verbose:                 verbose?,
            installed_as_dependency: true,
            force:                   false,
          ).install
        else
          fi = FormulaInstaller.new(
            cask_or_formula,
            **{
              show_header:             true,
              installed_as_dependency: true,
              installed_on_request:    false,
              verbose:                 verbose?,
            }.compact,
          )
          fi.prelude
          fi.fetch
          fi.install
          fi.finish
        end
      end
    end

    def caveats
      self.class.caveats(@cask)
    end

    def save_caskfile
      old_savedir = @cask.metadata_timestamped_path

      return unless @cask.sourcefile_path

      savedir = @cask.metadata_subdir("Casks", timestamp: :now, create: true)
      FileUtils.copy @cask.sourcefile_path, savedir
      old_savedir&.rmtree
    end

    def save_config_file
      @cask.config_path.atomic_write(@cask.config.to_json)
    end

    def uninstall
      oh1 "Uninstalling Cask #{Formatter.identifier(@cask)}"
      uninstall_artifacts(clear: true)
      remove_config_file if !reinstall? && !upgrade?
      purge_versioned_files
      purge_caskroom_path if force?
    end

    def remove_config_file
      FileUtils.rm_f @cask.config_path
      @cask.config_path.parent.rmdir_if_possible
    end

    def start_upgrade
      uninstall_artifacts
      backup
    end

    def backup
      @cask.staged_path.rename backup_path
      @cask.metadata_versioned_path.rename backup_metadata_path
    end

    def restore_backup
      return if !backup_path.directory? || !backup_metadata_path.directory?

      Pathname.new(@cask.staged_path).rmtree if @cask.staged_path.exist?
      Pathname.new(@cask.metadata_versioned_path).rmtree if @cask.metadata_versioned_path.exist?

      backup_path.rename @cask.staged_path
      backup_metadata_path.rename @cask.metadata_versioned_path
    end

    def revert_upgrade
      opoo "Reverting upgrade for Cask #{@cask}"
      restore_backup
      install_artifacts
    end

    def finalize_upgrade
      ohai "Purging files for version #{@cask.version} of Cask #{@cask}"

      purge_backed_up_versioned_files

      puts summary
    end

    def uninstall_artifacts(clear: false)
      artifacts = @cask.artifacts

      odebug "Uninstalling artifacts"
      odebug "#{artifacts.length} #{"artifact".pluralize(artifacts.length)} defined", artifacts

      artifacts.each do |artifact|
        if artifact.respond_to?(:uninstall_phase)
          odebug "Uninstalling artifact of class #{artifact.class}"
          artifact.uninstall_phase(
            command: @command, verbose: verbose?, skip: clear, force: force?, upgrade: upgrade?,
          )
        end

        next unless artifact.respond_to?(:post_uninstall_phase)

        odebug "Post-uninstalling artifact of class #{artifact.class}"
        artifact.post_uninstall_phase(
          command: @command, verbose: verbose?, skip: clear, force: force?, upgrade: upgrade?,
        )
      end
    end

    def zap
      ohai "Implied `brew uninstall --cask #{@cask}`"
      uninstall_artifacts
      if (zap_stanzas = @cask.artifacts.select { |a| a.is_a?(Artifact::Zap) }).empty?
        opoo "No zap stanza present for Cask '#{@cask}'"
      else
        ohai "Dispatching zap stanza"
        zap_stanzas.each do |stanza|
          stanza.zap_phase(command: @command, verbose: verbose?, force: force?)
        end
      end
      ohai "Removing all staged versions of Cask '#{@cask}'"
      purge_caskroom_path
    end

    def backup_path
      return if @cask.staged_path.nil?

      Pathname("#{@cask.staged_path}.upgrading")
    end

    def backup_metadata_path
      return if @cask.metadata_versioned_path.nil?

      Pathname("#{@cask.metadata_versioned_path}.upgrading")
    end

    def gain_permissions_remove(path)
      Utils.gain_permissions_remove(path, command: @command)
    end

    def purge_backed_up_versioned_files
      # versioned staged distribution
      gain_permissions_remove(backup_path) if backup_path&.exist?

      # Homebrew Cask metadata
      return unless backup_metadata_path.directory?

      backup_metadata_path.children.each do |subdir|
        gain_permissions_remove(subdir)
      end
      backup_metadata_path.rmdir_if_possible
    end

    def purge_versioned_files
      ohai "Purging files for version #{@cask.version} of Cask #{@cask}"

      # versioned staged distribution
      gain_permissions_remove(@cask.staged_path) if @cask.staged_path&.exist?

      # Homebrew Cask metadata
      if @cask.metadata_versioned_path.directory?
        @cask.metadata_versioned_path.children.each do |subdir|
          gain_permissions_remove(subdir)
        end

        @cask.metadata_versioned_path.rmdir_if_possible
      end
      @cask.metadata_master_container_path.rmdir_if_possible unless upgrade?

      # toplevel staged distribution
      @cask.caskroom_path.rmdir_if_possible unless upgrade?
    end

    def purge_caskroom_path
      odebug "Purging all staged versions of Cask #{@cask}"
      gain_permissions_remove(@cask.caskroom_path)
    end
  end
end
