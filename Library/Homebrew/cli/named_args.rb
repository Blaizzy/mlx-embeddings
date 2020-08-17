# frozen_string_literal: true

require "cask/cask_loader"
require "delegate"
require "formulary"

module Homebrew
  module CLI
    class NamedArgs < SimpleDelegator
      def initialize(*args, override_spec: nil, force_bottle: false, flags: [])
        @args = args
        @override_spec = override_spec
        @force_bottle = force_bottle
        @flags = flags

        super(@args)
      end

      def to_formulae
        @to_formulae ||= (downcased_unique_named - homebrew_tap_cask_names).map do |name|
          Formulary.factory(name, spec, force_bottle: @force_bottle, flags: @flags)
        end.uniq(&:name).freeze
      end

      def to_formulae_and_casks
        @to_formulae_and_casks ||= begin
          formulae_and_casks = []

          downcased_unique_named.each do |name|
            formulae_and_casks << Formulary.factory(name, spec)

            puts "Treating #{name} as a formula. For the cask, use homebrew/cask/#{name}" if cask_exists_with_ref name
          rescue FormulaUnavailableError
            begin
              formulae_and_casks << Cask::CaskLoader.load(name)
            rescue Cask::CaskUnavailableError
              raise "No available formula or cask with the name \"#{name}\""
            end
          end

          formulae_and_casks.freeze
        end
      end

      def to_resolved_formulae
        @to_resolved_formulae ||= (downcased_unique_named - homebrew_tap_cask_names).map do |name|
          Formulary.resolve(name, spec: spec(nil), force_bottle: @force_bottle, flags: @flags)
        end.uniq(&:name).freeze
      end

      def to_resolved_formulae_to_casks
        @to_resolved_formulae_to_casks ||= begin
          resolved_formulae = []
          casks = []

          downcased_unique_named.each do |name|
            resolved_formulae << Formulary.resolve(name, spec: spec(nil), force_bottle: @force_bottle, flags: @flags)

            puts "Treating #{name} as a formula. For the cask, use homebrew/cask/#{name}" if cask_exists_with_ref name
          rescue FormulaUnavailableError
            begin
              casks << Cask::CaskLoader.load(name)
            rescue Cask::CaskUnavailableError
              raise "No available formula or cask with the name \"#{name}\""
            end
          end

          [resolved_formulae.freeze, casks.freeze].freeze
        end
      end

      def to_formulae_paths
        @to_formulae_paths ||= (downcased_unique_named - homebrew_tap_cask_names).map do |name|
          Formulary.path(name)
        end.uniq.freeze
      end

      def to_casks
        @to_casks ||= downcased_unique_named.map(&Cask::CaskLoader.method(:load)).freeze
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

            puts "Treating #{name} as a keg. For the cask, use homebrew/cask/#{name}" if cask_exists_with_ref name
          rescue NoSuchKegError, FormulaUnavailableError
            begin
              casks << Cask::CaskLoader.load(name)
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

      def spec(default = :stable)
        @override_spec || default
      end

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

            unless (prefix = f.installed_prefix).directory?
              raise MultipleVersionsInstalledError, "#{rack.basename} has multiple installed versions"
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

      def cask_exists_with_ref(ref)
        Cask::CaskLoader.load ref
      rescue Cask::CaskUnavailableError
        false
      end
    end
  end
end
