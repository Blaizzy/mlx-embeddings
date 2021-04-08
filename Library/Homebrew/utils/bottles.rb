# typed: true
# frozen_string_literal: true

require "tab"

module Utils
  # Helper functions for bottles.
  #
  # @api private
  module Bottles
    class << self
      extend T::Sig

      def tag
        @tag ||= Tag.new(system: T.must(ENV["HOMEBREW_SYSTEM"]).downcase.to_sym,
                         arch:   T.must(ENV["HOMEBREW_PROCESSOR"]).downcase.to_sym)
      end

      def built_as?(f)
        return false unless f.latest_version_installed?

        tab = Tab.for_keg(f.latest_installed_prefix)
        tab.built_as_bottle
      end

      def file_outdated?(f, file)
        filename = file.basename.to_s
        return false if f.bottle.blank?

        bottle_ext, bottle_tag, = extname_tag_rebuild(filename)
        return false if bottle_ext.blank?
        return false if bottle_tag != tag.to_s

        bottle_url_ext, = extname_tag_rebuild(f.bottle.url)

        bottle_ext && bottle_url_ext && bottle_ext != bottle_url_ext
      end

      def extname_tag_rebuild(filename)
        HOMEBREW_BOTTLES_EXTNAME_REGEX.match(filename).to_a
      end

      def receipt_path(bottle_file)
        path = Utils.popen_read("tar", "-tzf", bottle_file).lines.map(&:chomp).find do |line|
          line =~ %r{.+/.+/INSTALL_RECEIPT.json}
        end
        raise "This bottle does not contain the file INSTALL_RECEIPT.json: #{bottle_file}" unless path

        path
      end

      def resolve_formula_names(bottle_file)
        receipt_file_path = receipt_path bottle_file
        receipt_file = Utils.popen_read("tar", "-xOzf", bottle_file, receipt_file_path)
        name = receipt_file_path.split("/").first
        tap = Tab.from_file_content(receipt_file, "#{bottle_file}/#{receipt_file_path}").tap

        full_name = if tap.nil? || tap.core_tap?
          name
        else
          "#{tap}/#{name}"
        end

        [name, full_name]
      end

      def resolve_version(bottle_file)
        PkgVersion.parse receipt_path(bottle_file).split("/").second
      end

      def formula_contents(bottle_file,
                           name: resolve_formula_names(bottle_file)[0])
        bottle_version = resolve_version bottle_file
        formula_path = "#{name}/#{bottle_version}/.brew/#{name}.rb"
        contents = Utils.popen_read "tar", "-xOzf", bottle_file, formula_path
        raise BottleFormulaUnavailableError.new(bottle_file, formula_path) unless $CHILD_STATUS.success?

        contents
      end
    end

    # Denotes the arch and OS of a bottle.
    class Tag
      extend T::Sig

      attr_reader :system, :arch

      sig { params(value: Symbol).returns(T.attached_class) }
      def self.from_symbol(value)
        @all_archs_regex ||= begin
          all_archs = Hardware::CPU::ALL_ARCHS.map(&:to_s)
          /
            ^((?<arch>#{Regexp.union(all_archs)})_)?
            (?<system>[\w.]+)$
          /x
        end
        match = @all_archs_regex.match(value.to_s)
        raise ArgumentError, "Invalid bottle tag symbol" unless match

        system = match[:system].to_sym
        arch = match[:arch]&.to_sym || :x86_64
        new(system: system, arch: arch)
      end

      sig { params(system: Symbol, arch: Symbol).void }
      def initialize(system:, arch:)
        @system = system
        @arch = arch
      end

      def ==(other)
        if other.is_a?(Symbol)
          to_sym == other
        else
          self.class == other.class && system == other.system && arch == other.arch
        end
      end

      sig { returns(Symbol) }
      def to_sym
        if macos? && arch == :x86_64
          system
        else
          "#{arch}_#{system}".to_sym
        end
      end

      sig { returns(String) }
      def to_s
        to_sym.to_s
      end

      sig { returns(OS::Mac::Version) }
      def to_macos_version
        @to_macos_version ||= OS::Mac::Version.from_symbol(system)
      end

      sig { returns(T::Boolean) }
      def linux?
        system == :linux
      end

      sig { returns(T::Boolean) }
      def macos?
        to_macos_version
        true
      rescue MacOSVersionError
        false
      end

      sig { returns(String) }
      def default_prefix
        if linux?
          HOMEBREW_LINUX_DEFAULT_PREFIX
        elsif arch == :arm64
          HOMEBREW_MACOS_ARM_DEFAULT_PREFIX
        else
          HOMEBREW_DEFAULT_PREFIX
        end
      end

      sig { returns(String) }
      def default_cellar
        if linux?
          Homebrew::DEFAULT_LINUX_CELLAR
        elsif arch == :arm64
          Homebrew::DEFAULT_MACOS_ARM_CELLAR
        else
          Homebrew::DEFAULT_MACOS_CELLAR
        end
      end
    end

    # Helper functions for bottles hosted on Bintray.
    module Bintray
      def self.package(formula_name)
        package_name = formula_name.to_s.dup
        package_name.tr!("+", "x")
        package_name.sub!(/(.)@(\d)/, "\\1:\\2") # Handle foo@1.2 style formulae.
        package_name
      end

      def self.repository(tap = nil)
        if tap.nil? || tap.core_tap?
          "bottles"
        else
          "bottles-#{tap.repo}"
        end
      end
    end

    # Collector for bottle specifications.
    class Collector
      extend T::Sig

      extend Forwardable

      def_delegators :@checksums, :keys, :[], :[]=, :key?, :each_key, :dig

      sig { void }
      def initialize
        @checksums = {}
      end

      sig {
        params(
          tag:   T.any(Symbol, Utils::Bottles::Tag),
          exact: T::Boolean,
        ).returns(
          T.nilable([Checksum, Symbol, T.any(Symbol, String)]),
        )
      }
      def fetch_checksum_for(tag, exact: false)
        tag = Utils::Bottles::Tag.from_symbol(tag) if tag.is_a?(Symbol)
        tag = find_matching_tag(tag, exact: exact)&.to_sym
        return self[tag][:checksum], tag, self[tag][:cellar] if tag
      end

      private

      def find_matching_tag(tag, exact: false)
        tag if key?(tag.to_sym)
      end
    end
  end
end

require "extend/os/bottles"
