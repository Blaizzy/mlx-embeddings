# typed: true
# frozen_string_literal: true

require "keg"
require "formula"

module Homebrew
  # Helper module for uninstalling kegs.
  #
  # @api private
  module Uninstall
    module_function

    def uninstall_kegs(kegs_by_rack, force: false, ignore_dependencies: false, named_args: [])
      handle_unsatisfied_dependents(kegs_by_rack,
        ignore_dependencies: ignore_dependencies,
        named_args:          named_args)
      return if Homebrew.failed?

      kegs_by_rack.each do |rack, kegs|
        if force
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

  #   def perform_preinstall_checks(all_fatal: false, cc: nil)
  #     check_cpu
  #     attempt_directory_creation
  #     check_cc_argv(cc)
  #     Diagnostic.checks(:supported_configuration_checks, fatal: all_fatal)
  #     Diagnostic.checks(:fatal_preinstall_checks)
  #   end
  #   alias generic_perform_preinstall_checks perform_preinstall_checks
  #   module_function :generic_perform_preinstall_checks

  #   def perform_build_from_source_checks(all_fatal: false)
  #     Diagnostic.checks(:fatal_build_from_source_checks)
  #     Diagnostic.checks(:build_from_source_checks, fatal: all_fatal)
  #   end

  #   def check_cpu
  #     return if Hardware::CPU.intel? && Hardware::CPU.is_64_bit?

  #     message = "Sorry, Homebrew does not support your computer's CPU architecture!"
  #     if Hardware::CPU.arm?
  #       opoo message
  #       return
  #     elsif Hardware::CPU.ppc?
  #       message += <<~EOS
  #         For PowerPC Mac (PPC32/PPC64BE) support, see:
  #           #{Formatter.url("https://github.com/mistydemeo/tigerbrew")}
  #       EOS
  #     end
  #     abort message
  #   end
  #   private_class_method :check_cpu

  #   def attempt_directory_creation
  #     Keg::MUST_EXIST_DIRECTORIES.each do |dir|
  #       FileUtils.mkdir_p(dir) unless dir.exist?

  #       # Create these files to ensure that these directories aren't removed
  #       # by the Catalina installer.
  #       # (https://github.com/Homebrew/brew/issues/6263)
  #       keep_file = dir/".keepme"
  #       FileUtils.touch(keep_file) unless keep_file.exist?
  #     rescue
  #       nil
  #     end
  #   end
  #   private_class_method :attempt_directory_creation

  #   def check_cc_argv(cc)
  #     return unless cc

  #     @checks ||= Diagnostic::Checks.new
  #     opoo <<~EOS
  #       You passed `--cc=#{cc}`.
  #       #{@checks.please_create_pull_requests}
  #     EOS
  #   end
  #   private_class_method :check_cc_argv
  end
end
