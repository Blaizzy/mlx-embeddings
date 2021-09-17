# frozen_string_literal: true

module Unparser
  # Validation of unparser results
  class Validation
    include Adamantium, Anima.new(
      :generated_node,
      :generated_source,
      :identification,
      :original_node,
      :original_source
    )

    # Test if source could be unparsed successfully
    #
    # @return [Boolean]
    #
    # @api private
    #
    def success?
      [
        original_source,
        original_node,
        generated_source,
        generated_node
      ].all?(&:right?) && generated_node.from_right.==(original_node.from_right)
    end

    # Return error report
    #
    # @return [String]
    #
    # @api private
    #
    def report
      message = [identification]

      message.concat(make_report('Original-Source',  :original_source))
      message.concat(make_report('Generated-Source', :generated_source))
      message.concat(make_report('Original-Node',    :original_node))
      message.concat(make_report('Generated-Node',   :generated_node))
      message.concat(node_diff_report)

      message.join("\n")
    end
    memoize :report

    # Create validator from string
    #
    # @param [String] original_source
    #
    # @return [Validator]
    def self.from_string(original_source)
      original_node = Unparser
        .parse_either(original_source)

      generated_source = original_node
        .lmap(&method(:const_unit))
        .bind(&Unparser.method(:unparse_either))

      generated_node = generated_source
        .lmap(&method(:const_unit))
        .bind(&Unparser.method(:parse_either))

      new(
        identification:   '(string)',
        original_source:  Either::Right.new(original_source),
        original_node:    original_node,
        generated_source: generated_source,
        generated_node:   generated_node
      )
    end

    # Create validator from node
    #
    # @param [Parser::AST::Node] original_node
    #
    # @return [Validator]
    def self.from_node(original_node)
      generated_source = Unparser.unparse_either(original_node)

      generated_node = generated_source
        .lmap(&method(:const_unit))
        .bind(&Unparser.public_method(:parse_either))

      new(
        identification:   '(string)',
        original_source:  generated_source,
        original_node:    Either::Right.new(original_node),
        generated_source: generated_source,
        generated_node:   generated_node
      )
    end

    # Create validator from file
    #
    # @param [Pathname] path
    #
    # @return [Validator]
    def self.from_path(path)
      from_string(path.read).with(identification: path.to_s)
    end

  private

    def make_report(label, attribute_name)
      ["#{label}:"].concat(public_send(attribute_name).either(method(:report_exception), ->(value) { [value] }))
    end

    def report_exception(exception)
      if exception
        [exception.inspect].concat(exception.backtrace.take(20))
      else
        ['undefined']
      end
    end

    def node_diff_report
      diff = nil

      original_node.fmap do |original|
        generated_node.fmap do |generated|
          diff = Diff.new(
            original.to_s.lines.map(&:chomp),
            generated.to_s.lines.map(&:chomp)
          ).colorized_diff
        end
      end

      diff ? ['Node-Diff:', diff] : []
    end

    def self.const_unit(_value); end
    private_class_method :const_unit

    class Literal < self
      def success?
        original_source.eql?(generated_source)
      end

      def report
        message = [identification]

        message.concat(make_report('Original-Source',  :original_source))
        message.concat(make_report('Generated-Source', :generated_source))
        message.concat(make_report('Original-Node',    :original_node))
        message.concat(make_report('Generated-Node',   :generated_node))
        message.concat(node_diff_report)
        message.concat(source_diff_report)

        message.join("\n")
      end

    private

      def source_diff_report
        diff = nil

        original_source.fmap do |original|
          generated_source.fmap do |generated|
            diff = Diff.new(
              original.split("\n", -1),
              generated.split("\n", -1)
            ).colorized_diff
          end
        end

        diff ? ['Source-Diff:', diff] : []
      end
    end # Literal
  end # Validation
end # Unparser
