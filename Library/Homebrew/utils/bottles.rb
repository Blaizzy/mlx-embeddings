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
        return unless f.bottle && filename.match(Pathname::BOTTLE_EXTNAME_RX)

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

      def add_bottle_stanza!(formula_contents, bottle_output)
        Homebrew.install_bundler_gems!
        require "rubocop-ast"

        ruby_version = Version.new(HOMEBREW_REQUIRED_RUBY_VERSION).major_minor.to_f
        processed_source = RuboCop::AST::ProcessedSource.new(formula_contents, ruby_version)
        root_node = processed_source.ast

        class_node = if root_node.class_type?
          root_node
        elsif root_node.begin_type?
          root_node.children.find do |n|
            n.class_type? && n.parent_class&.const_name == "Formula"
          end
        end

        odie "Could not find formula class!" if class_node.nil?

        body_node = class_node.body
        odie "Formula class is empty!" if body_node.nil?

        node_before_bottle = if body_node.begin_type?
          body_node.children.compact.reduce do |previous_child, current_child|
            break previous_child unless component_before_bottle_block? current_child

            current_child
          end
        else
          body_node
        end
        node_before_bottle = node_before_bottle.last_argument if node_before_bottle.send_type?

        expr_before_bottle = node_before_bottle.location.expression
        processed_source.comments.each do |comment|
          comment_expr = comment.location.expression
          distance = comment_expr.first_line - expr_before_bottle.first_line
          case distance
          when 0
            if comment_expr.last_line > expr_before_bottle.last_line ||
               comment_expr.end_pos > expr_before_bottle.end_pos
              expr_before_bottle = comment_expr
            end
          when 1
            expr_before_bottle = comment_expr
          end
        end

        tree_rewriter = Parser::Source::TreeRewriter.new(processed_source.buffer)
        tree_rewriter.insert_after(expr_before_bottle, "\n\n#{bottle_output.chomp}")
        formula_contents.replace(tree_rewriter.process)
      end

      private

      def component_before_bottle_block?(node)
        require "rubocops/components_order"

        RuboCop::Cop::FormulaAudit::ComponentsOrder::COMPONENT_PRECEDENCE_LIST.each do |components|
          components.each do |component|
            return false if component[:name] == :bottle && component[:type] == :block_call

            case component[:type]
            when :method_call
              return true if node.send_type? && node.method_name == component[:name]
            when :block_call
              return true if node.block_type? && node.method_name == component[:name]
            end
          end
        end
        false
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

      def_delegators :@checksums, :keys, :[], :[]=, :key?, :each_key

      sig { void }
      def initialize
        @checksums = {}
      end

      def fetch_checksum_for(tag)
        tag = find_matching_tag(tag)
        return self[tag], tag if tag
      end

      private

      def find_matching_tag(tag)
        tag if key?(tag)
      end
    end
  end
end

require "extend/os/bottles"
