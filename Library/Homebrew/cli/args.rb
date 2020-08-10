# frozen_string_literal: true

require "ostruct"

module Homebrew
  module CLI
    class Args < OpenStruct
      attr_reader :options_only, :flags_only

      # undefine tap to allow --tap argument
      undef tap

      def initialize
        super()

        @processed_options = []
        @options_only = []
        @flags_only = []

        # Can set these because they will be overwritten by freeze_named_args!
        # (whereas other values below will only be overwritten if passed).
        self[:named_args] = []
        self[:remaining] = []
      end

      def freeze_remaining_args!(remaining_args)
        self[:remaining] = remaining_args.freeze
      end

      def freeze_named_args!(named_args)
        # Reset cache values reliant on named_args
        @formulae = nil
        @formulae_and_casks = nil
        @resolved_formulae = nil
        @resolved_formulae_casks = nil
        @formulae_paths = nil
        @casks = nil
        @loaded_casks = nil
        @kegs = nil
        @kegs_casks = nil

        self[:named_args] = named_args.freeze
      end

      def freeze_processed_options!(processed_options)
        # Reset cache values reliant on processed_options
        @cli_args = nil

        @processed_options += processed_options
        @processed_options.freeze

        @options_only = cli_args.select { |a| a.start_with?("-") }.freeze
        @flags_only = cli_args.select { |a| a.start_with?("--") }.freeze
      end

      def named
        named_args || []
      end

      def no_named?
        named.blank?
      end

      def formulae
        require "formula"

        @formulae ||= (downcased_unique_named - casks).map do |name|
          Formulary.factory(name, spec, force_bottle: force_bottle?, flags: flags_only)
        end.uniq(&:name).freeze
      end

      def formulae_and_casks
        @formulae_and_casks ||= begin
          formulae_and_casks = []

          downcased_unique_named.each do |name|
            formulae_and_casks << Formulary.factory(name, spec)
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

      def resolved_formulae
        require "formula"

        @resolved_formulae ||= (downcased_unique_named - casks).map do |name|
          Formulary.resolve(name, spec: spec(nil), force_bottle: force_bottle?, flags: flags_only)
        end.uniq(&:name).freeze
      end

      def resolved_formulae_casks
        @resolved_formulae_casks ||= begin
          resolved_formulae = []
          casks = []

          downcased_unique_named.each do |name|
            resolved_formulae << Formulary.resolve(name, spec: spec(nil),
                                                   force_bottle: force_bottle?, flags: flags_only)
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

      def formulae_paths
        @formulae_paths ||= (downcased_unique_named - casks).map do |name|
          Formulary.path(name)
        end.uniq.freeze
      end

      def casks
        @casks ||= downcased_unique_named.grep(HOMEBREW_CASK_TAP_CASK_REGEX)
                                         .freeze
      end

      def loaded_casks
        @loaded_casks ||= downcased_unique_named.map(&Cask::CaskLoader.method(:load)).freeze
      end

      def kegs
        @kegs ||= downcased_unique_named.map do |name|
          resolve_keg name
        rescue NoSuchKegError => e
          if (reason = Homebrew::MissingFormula.suggest_command(name, "uninstall"))
            $stderr.puts reason
          end
          raise e
        end.freeze
      end

      def kegs_casks
        @kegs_casks ||= begin
          kegs = []
          casks = []

          downcased_unique_named.each do |name|
            kegs << resolve_keg(name)
          rescue NoSuchKegError
            begin
              casks << Cask::CaskLoader.load(name)
            rescue Cask::CaskUnavailableError
              raise "No installed keg or cask with the name \"#{name}\""
            end
          end

          [kegs.freeze, casks.freeze].freeze
        end
      end

      def build_stable?
        !(HEAD? || devel?)
      end

      def build_from_source_formulae
        if build_from_source? || build_bottle?
          formulae.map(&:full_name)
        else
          []
        end
      end

      def include_test_formulae
        if include_test?
          formulae.map(&:full_name)
        else
          []
        end
      end

      def value(name)
        arg_prefix = "--#{name}="
        flag_with_value = flags_only.find { |arg| arg.start_with?(arg_prefix) }
        return unless flag_with_value

        flag_with_value.delete_prefix(arg_prefix)
      end

      def context
        Context::ContextStruct.new(debug: debug?, quiet: quiet?, verbose: verbose?)
      end

      private

      def option_to_name(option)
        option.sub(/\A--?/, "")
              .tr("-", "_")
      end

      def cli_args
        return @cli_args if @cli_args

        @cli_args = []
        @processed_options.each do |short, long|
          option = long || short
          switch = "#{option_to_name(option)}?".to_sym
          flag = option_to_name(option).to_sym
          if @table[switch] == true || @table[flag] == true
            @cli_args << option
          elsif @table[flag].instance_of? String
            @cli_args << option + "=" + @table[flag]
          elsif @table[flag].instance_of? Array
            @cli_args << option + "=" + @table[flag].join(",")
          end
        end
        @cli_args.freeze
      end

      def downcased_unique_named
        # Only lowercase names, not paths, bottle filenames or URLs
        named.map do |arg|
          if arg.include?("/") || arg.end_with?(".tar.gz") || File.exist?(arg)
            arg
          else
            arg.downcase
          end
        end.uniq
      end

      def spec(default = :stable)
        if HEAD?
          :head
        elsif devel?
          :devel
        else
          default
        end
      end

      def resolve_keg(name)
        require "keg"
        require "formula"
        require "missing_formula"

        raise UsageError if name.blank?

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
    end
  end
end
