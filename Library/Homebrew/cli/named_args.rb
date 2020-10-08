# frozen_string_literal: true

require "delegate"

require "cask/cask_loader"
require "cli/args"
require "formulary"
require "missing_formula"

module Homebrew
  module CLI
    # Helper class for loading formulae/casks from named arguments.
    #
    # @api private
    class NamedArgs < SimpleDelegator
      def initialize(*args, parent: Args.new, override_spec: nil, force_bottle: false, flags: [])
        @args = args
        @override_spec = override_spec
        @force_bottle = force_bottle
        @flags = flags
        @parent = parent

        super(@args)
      end

      def to_casks
        @to_casks ||= to_formulae_and_casks(only: :cask).freeze
      end

      def to_formulae
        @to_formulae ||= to_formulae_and_casks(only: :formula).freeze
      end

      def to_formulae_and_casks(only: nil, method: nil)
        @to_formulae_and_casks ||= {}
        @to_formulae_and_casks[only] ||= begin
          to_objects(only: only, method: method).reject { |o| o.is_a?(Tap) }.freeze
        end
      end

      def load_formula_or_cask(name, only: nil, method: nil)
        if only != :cask
          begin
            formula = case method
            when nil, :factory
              Formulary.factory(name, *spec, force_bottle: @force_bottle, flags: @flags)
            when :resolve
              Formulary.resolve(name, spec: spec, force_bottle: @force_bottle, flags: @flags)
            else
              raise
            end

            warn_if_cask_conflicts(name, "formula") unless only == :formula
            return formula
          rescue FormulaUnavailableError => e
            raise e if only == :formula
          end
        end

        if only != :formula
          begin
            return Cask::CaskLoader.load(name, config: Cask::Config.from_args(@parent))
          rescue Cask::CaskUnavailableError => e
            raise e if only == :cask
          end
        end

        raise FormulaOrCaskUnavailableError, name
      end
      private :load_formula_or_cask

      def resolve_formula(name)
        Formulary.resolve(name, spec: spec, force_bottle: @force_bottle, flags: @flags)
      end
      private :resolve_formula

      def to_resolved_formulae
        @to_resolved_formulae ||= to_formulae_and_casks(only: :formula, method: :resolve)
                                  .freeze
      end

      def to_resolved_formulae_to_casks(only: nil)
        @to_resolved_formulae_to_casks ||= to_formulae_and_casks(method: :resolve, only: only)
                                           .partition { |o| o.is_a?(Formula) }
                                           .map(&:freeze).freeze
      end

      # Convert named arguments to `Tap`, `Formula` or `Cask` objects.
      # If both a formula and cask exist with the same name, returns the
      # formula and prints a warning unless `only` is specified.
      def to_objects(only: nil, method: nil)
        @to_objects ||= {}
        @to_objects[only] ||= downcased_unique_named.flat_map do |name|
          next Tap.fetch(name) if only == :tap || (only.nil? && name.count("/") == 1 && !name.start_with?("./", "/"))

          load_formula_or_cask(name, only: only, method: method)
        end.uniq.freeze
      end
      private :to_objects

      def to_formulae_paths
        to_paths(only: :formulae)
      end

      # Keep existing paths and try to convert others to tap, formula or cask paths.
      # If a cask and formula with the same name exist, includes both their paths
      # unless `only` is specified.
      def to_paths(only: nil)
        @to_paths ||= {}
        @to_paths[only] ||= downcased_unique_named.flat_map do |name|
          if File.exist?(name)
            Pathname(name)
          elsif name.count("/") == 1
            Tap.fetch(name).path
          else
            next Formulary.path(name) if only == :formulae
            next Cask::CaskLoader.path(name) if only == :casks

            formula_path = Formulary.path(name)
            cask_path = Cask::CaskLoader.path(name)

            paths = []

            paths << formula_path if formula_path.exist?
            paths << cask_path if cask_path.exist?

            paths.empty? ? name : paths
          end
        end.uniq.freeze
      end

      def to_kegs
        @to_kegs ||= downcased_unique_named.map do |name|
          resolve_keg name
        rescue NoSuchKegError => e
          if (reason = Homebrew::MissingFormula.suggest_command(name, "uninstall"))
            $stderr.puts reason
          end
          raise e
        end.freeze
      end

      def to_kegs_to_casks
        @to_kegs_to_casks ||= begin
          kegs = []
          casks = []

          downcased_unique_named.each do |name|
            kegs << resolve_keg(name)

            warn_if_cask_conflicts(name, "keg")
          rescue NoSuchKegError, FormulaUnavailableError
            begin
              casks << Cask::CaskLoader.load(name, config: Cask::Config.from_args(@parent))
            rescue Cask::CaskUnavailableError
              raise "No installed keg or cask with the name \"#{name}\""
            end
          end

          [kegs.freeze, casks.freeze].freeze
        end
      end

      def homebrew_tap_cask_names
        downcased_unique_named.grep(HOMEBREW_CASK_TAP_CASK_REGEX)
      end

      private

      def downcased_unique_named
        # Only lowercase names, not paths, bottle filenames or URLs
        map do |arg|
          if arg.include?("/") || arg.end_with?(".tar.gz") || File.exist?(arg)
            arg
          else
            arg.downcase
          end
        end.uniq
      end

      def spec
        @override_spec
      end
      private :spec

      def resolve_keg(name)
        raise UsageError if name.blank?

        require "keg"

        rack = Formulary.to_rack(name.downcase)

        dirs = rack.directory? ? rack.subdirs : []
        raise NoSuchKegError, rack.basename if dirs.empty?

        linked_keg_ref = HOMEBREW_LINKED_KEGS/rack.basename
        opt_prefix = HOMEBREW_PREFIX/"opt/#{rack.basename}"

        begin
          if opt_prefix.symlink? && opt_prefix.directory?
            Keg.new(opt_prefix.resolved_path)
          elsif linked_keg_ref.symlink? && linked_keg_ref.directory?
            Keg.new(linked_keg_ref.resolved_path)
          elsif dirs.length == 1
            Keg.new(dirs.first)
          else
            f = if name.include?("/") || File.exist?(name)
              Formulary.factory(name)
            else
              Formulary.from_rack(rack)
            end

            unless (prefix = f.latest_installed_prefix).directory?
              raise MultipleVersionsInstalledError, <<~EOS
                #{rack.basename} has multiple installed versions
                Run `brew uninstall --force #{rack.basename}` to remove all versions.
              EOS
            end

            Keg.new(prefix)
          end
        rescue FormulaUnavailableError
          raise MultipleVersionsInstalledError, <<~EOS
            Multiple kegs installed to #{rack}
            However we don't know which one you refer to.
            Please delete (with rm -rf!) all but one and then try again.
          EOS
        end
      end

      def warn_if_cask_conflicts(ref, loaded_type)
        cask = Cask::CaskLoader.load ref
        message = "Treating #{ref} as a #{loaded_type}."
        message += " For the cask, use #{cask.tap.name}/#{cask.token}" if cask.tap.present?
        opoo message.freeze
      rescue Cask::CaskUnavailableError
        # No ref conflict with a cask, do nothing
      end
    end
  end
end
