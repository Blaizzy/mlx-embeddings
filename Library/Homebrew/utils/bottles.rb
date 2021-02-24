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
        @tag ||= "#{ENV["HOMEBREW_PROCESSOR"]}_#{ENV["HOMEBREW_SYSTEM"]}".downcase.to_sym
      end

      def built_as?(f)
        return false unless f.latest_version_installed?

        tab = Tab.for_keg(f.latest_installed_prefix)
        tab.built_as_bottle
      end

      def file_outdated?(f, file)
        filename = file.basename.to_s
        return if f.bottle.blank? || !filename.match?(Pathname::BOTTLE_EXTNAME_RX)

        bottle_ext = filename[native_regex, 1]
        bottle_url_ext = f.bottle.url[native_regex, 1]

        bottle_ext && bottle_url_ext && bottle_ext != bottle_url_ext
      end

      sig { returns(Regexp) }
      def native_regex
        /(\.#{Regexp.escape(tag.to_s)}\.bottle\.(\d+\.)?tar\.gz)$/o
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

      sig { params(tag: Symbol, exact: T::Boolean).returns(T.nilable([Checksum, Symbol, T.any(Symbol, String)])) }
      def fetch_checksum_for(tag, exact: false)
        tag = find_matching_tag(tag, exact: exact)
        return self[tag][:checksum], tag, self[tag][:cellar] if tag
      end

      private

      def find_matching_tag(tag, exact: false)
        tag if key?(tag)
      end
    end
  end
end

require "extend/os/bottles"
