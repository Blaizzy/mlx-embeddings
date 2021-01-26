# typed: false
# frozen_string_literal: true

require "delegate"

require "requirements/macos_requirement"

module Cask
  class DSL
    # Class corresponding to the `depends_on` stanza.
    #
    # @api private
    class DependsOn < SimpleDelegator
      VALID_KEYS = Set.new([
                             :formula,
                             :cask,
                             :macos,
                             :arch,
                             :x11,
                             :java,
                           ]).freeze

      VALID_ARCHES = {
        intel:  { type: :intel, bits: 64 },
        # specific
        x86_64: { type: :intel, bits: 64 },
        arm64:  { type: :arm, bits: 64 },
      }.freeze

      attr_reader :arch, :cask, :formula, :java, :macos, :x11

      def initialize
        super({})
        @cask ||= []
        @formula ||= []
      end

      def load(**pairs)
        pairs.each do |key, value|
          raise "invalid depends_on key: '#{key.inspect}'" unless VALID_KEYS.include?(key)

          self[key] = send(:"#{key}=", *value)
        end
      end

      def formula=(*args)
        @formula.concat(args)
      end

      def cask=(*args)
        @cask.concat(args)
      end

      def macos=(*args)
        raise "Only a single 'depends_on macos' is allowed." if defined?(@macos)

        begin
          @macos = if args.count > 1
            MacOSRequirement.new([args], comparator: "==")
          elsif MacOS::Version::SYMBOLS.key?(args.first)
            MacOSRequirement.new([args.first], comparator: "==")
          elsif /^\s*(?<comparator><|>|[=<>]=)\s*:(?<version>\S+)\s*$/ =~ args.first
            MacOSRequirement.new([version.to_sym], comparator: comparator)
          elsif /^\s*(?<comparator><|>|[=<>]=)\s*(?<version>\S+)\s*$/ =~ args.first
            MacOSRequirement.new([version], comparator: comparator)
          else # rubocop:disable Lint/DuplicateBranch
            MacOSRequirement.new([args.first], comparator: "==")
          end
        rescue MacOSVersionError => e
          raise "invalid 'depends_on macos' value: #{e}"
        end
      end

      def arch=(*args)
        @arch ||= []
        arches = args.map do |elt|
          elt.to_s.downcase.sub(/^:/, "").tr("-", "_").to_sym
        end
        invalid_arches = arches - VALID_ARCHES.keys
        raise "invalid 'depends_on arch' values: #{invalid_arches.inspect}" unless invalid_arches.empty?

        @arch.concat(arches.map { |arch| VALID_ARCHES[arch] })
      end

      def java=(arg)
        odeprecated "depends_on :java", "depends_on a specific Java formula"

        @java = arg
      end

      def x11=(arg)
        raise "invalid 'depends_on x11' value: #{arg.inspect}" unless [true, false].include?(arg)

        odeprecated "depends_on :x11", "depends_on specific X11 formula(e)"

        @x11 = arg
      end
    end
  end
end
