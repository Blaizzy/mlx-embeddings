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

      def tag(symbol = nil)
        return Tag.from_symbol(symbol) if symbol.present?

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
        bottle_file_list(bottle_file).find do |line|
          line =~ %r{.+/.+/INSTALL_RECEIPT.json}
        end
      end

      def file_from_bottle(bottle_file, file_path)
        Utils.popen_read("tar", "--extract", "--to-stdout", "--file", bottle_file, file_path)
      end

      def resolve_formula_names(bottle_file)
        name = bottle_file_list(bottle_file).first.to_s.split("/").first
        full_name = if (receipt_file_path = receipt_path(bottle_file))
          receipt_file = file_from_bottle(bottle_file, receipt_file_path)
          tap = Tab.from_file_content(receipt_file, "#{bottle_file}/#{receipt_file_path}").tap
          "#{tap}/#{name}" if tap.present? && !tap.core_tap?
        elsif (bottle_json_path = Pathname(bottle_file.sub(/\.tar\.gz$/, ".json"))) &&
              bottle_json_path.exist? &&
              (bottle_json_path_contents = bottle_json_path.read.presence) &&
              (bottle_json = JSON.parse(bottle_json_path_contents).presence) &&
              bottle_json.is_a?(Hash)
          bottle_json.keys.first.presence
        end
        full_name ||= name

        [name, full_name]
      end

      def resolve_version(bottle_file)
        version = bottle_file_list(bottle_file).first.to_s.split("/").second
        PkgVersion.parse(version)
      end

      def formula_contents(bottle_file,
                           name: resolve_formula_names(bottle_file)[0])
        bottle_version = resolve_version bottle_file
        formula_path = "#{name}/#{bottle_version}/.brew/#{name}.rb"
        contents = file_from_bottle(bottle_file, formula_path)
        raise BottleFormulaUnavailableError.new(bottle_file, formula_path) unless $CHILD_STATUS.success?

        contents
      end

      def path_resolved_basename(root_url, name, checksum, filename)
        if root_url.match?(GitHubPackages::URL_REGEX)
          image_name = GitHubPackages.image_formula_name(name)
          ["#{image_name}/blobs/sha256:#{checksum}", filename&.github_packages]
        else
          filename&.url_encode
        end
      end

      def load_tab(formula)
        keg = Keg.new(formula.prefix)
        tabfile = keg/Tab::FILENAME
        bottle_json_path = formula.local_bottle_path&.sub(/\.tar\.gz$/, ".json")

        if (tab_attributes = formula.bottle_tab_attributes.presence)
          Tab.from_file_content(tab_attributes.to_json, tabfile)
        elsif !tabfile.exist? && bottle_json_path&.exist?
          _, tag, = Utils::Bottles.extname_tag_rebuild(formula.local_bottle_path)
          bottle_hash = JSON.parse(File.read(bottle_json_path))
          tab_json = bottle_hash[formula.full_name]["bottle"]["tags"][tag]["tab"].to_json
          Tab.from_file_content(tab_json, tabfile)
        else
          Tab.for_keg(keg)
        end
      end

      private

      def bottle_file_list(bottle_file)
        @bottle_file_list ||= {}
        @bottle_file_list[bottle_file] ||= Utils.popen_read("tar", "--list", "--file", bottle_file)
                                                .lines
                                                .map(&:chomp)
      end
    end

    # Denotes the arch and OS of a bottle.
    class Tag
      extend T::Sig

      attr_reader :system, :arch

      sig { params(value: Symbol).returns(T.attached_class) }
      def self.from_symbol(value)
        return new(system: :all, arch: :all) if value == :all

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
        if system == :all && arch == :all
          :all
        elsif macos? && arch == :x86_64
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
          tag:               T.any(Symbol, Utils::Bottles::Tag),
          no_older_versions: T::Boolean,
        ).returns(
          T.nilable([Checksum, Symbol, T.any(Symbol, String)]),
        )
      }
      def fetch_checksum_for(tag, no_older_versions: false)
        tag = Utils::Bottles::Tag.from_symbol(tag) if tag.is_a?(Symbol)
        tag = find_matching_tag(tag, no_older_versions: no_older_versions)&.to_sym
        return self[tag][:checksum], tag, self[tag][:cellar] if tag
      end

      private

      def find_matching_tag(tag, no_older_versions: false)
        if key?(tag.to_sym)
          tag
        elsif key?(:all)
          Tag.from_symbol(:all)
        end
      end
    end
  end
end

require "extend/os/bottles"
