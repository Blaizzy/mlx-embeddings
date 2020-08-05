# frozen_string_literal: true

require "keg"
require "formula"
require "diagnostic"
require "migrator"
require "cli/parser"
require "cask/all"
require "cask/cmd"
require "cask/cask_loader"

module Homebrew
  module_function

  def uninstall_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `uninstall`, `rm`, `remove` [<options>] <formula>

        Uninstall <formula>.
      EOS
      switch "-f", "--force",
             description: "Delete all installed versions of <formula>."
      switch "--ignore-dependencies",
             description: "Don't fail uninstall, even if <formula> is a dependency of any installed "\
                          "formulae."

      min_named :formula
    end
  end

  def uninstall
    args = uninstall_args.parse

    if args.force?
      casks = []
      kegs_by_rack = {}

      args.named.each do |name|
        rack = Formulary.to_rack(name)

        if rack.directory?
          kegs_by_rack[rack] = rack.subdirs.map { |d| Keg.new(d) }
        else
          begin
            casks << Cask::CaskLoader.load(name)
          rescue Cask::CaskUnavailableError
            # Since the uninstall was forced, ignore any unavailable casks
          end
        end
      end
    else
      all_kegs, casks = args.kegs_casks
      kegs_by_rack = all_kegs.group_by(&:rack)
    end

    handle_unsatisfied_dependents(kegs_by_rack,
                                  ignore_dependencies: args.ignore_dependencies?,
                                  named_args:          args.named)
    return if Homebrew.failed?

    kegs_by_rack.each do |rack, kegs|
      if args.force?
        name = rack.basename

        if rack.directory?
          puts "Uninstalling #{name}... (#{rack.abv})"
          kegs.each do |keg|
            keg.unlink
            keg.uninstall
          end
        end

        rm_pin rack
      else
        kegs.each do |keg|
          begin
            f = Formulary.from_rack(rack)
            if f.pinned?
              onoe "#{f.full_name} is pinned. You must unpin it to uninstall."
              next
            end
          rescue
            nil
          end

          keg.lock do
            puts "Uninstalling #{keg}... (#{keg.abv})"
            keg.unlink
            keg.uninstall
            rack = keg.rack
            rm_pin rack

            if rack.directory?
              versions = rack.subdirs.map(&:basename)
              puts "#{keg.name} #{versions.to_sentence} #{"is".pluralize(versions.count)} still installed."
              puts "Run `brew uninstall --force #{keg.name}` to remove all versions."
            end

            next unless f

            paths = f.pkgetc.find.map(&:to_s) if f.pkgetc.exist?
            if paths.present?
              puts
              opoo <<~EOS
                The following #{f.name} configuration files have not been removed!
                If desired, remove them manually with `rm -rf`:
                  #{paths.sort.uniq.join("\n  ")}
              EOS
            end

            unversioned_name = f.name.gsub(/@.+$/, "")
            maybe_paths = Dir.glob("#{f.etc}/*#{unversioned_name}*")
            maybe_paths -= paths if paths.present?
            if maybe_paths.present?
              puts
              opoo <<~EOS
                The following may be #{f.name} configuration files and have not been removed!
                If desired, remove them manually with `rm -rf`:
                  #{maybe_paths.sort.uniq.join("\n  ")}
              EOS
            end
          end
        end
      end
    end

    return if casks.blank?

    Cask::Cmd::Uninstall.uninstall_casks(
      *casks,
      binaries: args.binaries?,
      verbose:  args.verbose?,
      force:    args.force?,
    )
  rescue MultipleVersionsInstalledError => e
    ofail e
    puts "Run `brew uninstall --force #{e.name}` to remove all versions."
  ensure
    # If we delete Cellar/newname, then Cellar/oldname symlink
    # can become broken and we have to remove it.
    if HOMEBREW_CELLAR.directory?
      HOMEBREW_CELLAR.children.each do |rack|
        rack.unlink if rack.symlink? && !rack.resolved_path_exists?
      end
    end
  end

  def handle_unsatisfied_dependents(kegs_by_rack, ignore_dependencies: false, named_args: [])
    return if ignore_dependencies

    all_kegs = kegs_by_rack.values.flatten(1)
    check_for_dependents(all_kegs, named_args: named_args)
  rescue MethodDeprecatedError
    # Silently ignore deprecations when uninstalling.
    nil
  end

  def check_for_dependents(kegs, named_args: [])
    return false unless result = Keg.find_some_installed_dependents(kegs)

    if Homebrew::EnvConfig.developer?
      DeveloperDependentsMessage.new(*result, named_args: named_args).output
    else
      NondeveloperDependentsMessage.new(*result, named_args: named_args).output
    end

    true
  end

  class DependentsMessage
    attr_reader :reqs, :deps, :named_args

    def initialize(requireds, dependents, named_args: [])
      @reqs = requireds
      @deps = dependents
      @named_args = named_args
    end

    protected

    def sample_command
      "brew uninstall --ignore-dependencies #{named_args.join(" ")}"
    end

    def are_required_by_deps
      "#{"is".pluralize(reqs.count)} required by #{deps.to_sentence}, " \
      "which #{"is".pluralize(deps.count)} currently installed"
    end
  end

  class DeveloperDependentsMessage < DependentsMessage
    def output
      opoo <<~EOS
        #{reqs.to_sentence} #{are_required_by_deps}.
        You can silence this warning with:
          #{sample_command}
      EOS
    end
  end

  class NondeveloperDependentsMessage < DependentsMessage
    def output
      ofail <<~EOS
        Refusing to uninstall #{reqs.to_sentence}
        because #{"it".pluralize(reqs.count)} #{are_required_by_deps}.
        You can override this and force removal with:
          #{sample_command}
      EOS
    end
  end

  def rm_pin(rack)
    Formulary.from_rack(rack).unpin
  rescue
    nil
  end
end
