# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits deprecate! date and disable! date
      class DeprecateDisableDate < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          [:deprecate!, :disable!].each do |method|
            node = find_node_method_by_name(body_node, method)

            next if node.nil?

            date(node) do |date_node|
              Date.iso8601(string_content(date_node))
            rescue ArgumentError
              fixed_date_string = Date.parse(string_content(date_node)).iso8601
              offending_node(date_node)
              problem "Use `#{fixed_date_string}` to comply with ISO 8601"
            end
          end
        end

        def autocorrect(node)
          lambda do |corrector|
            fixed_fixed_date_string = Date.parse(string_content(node)).iso8601
            corrector.replace(node.source_range, "\"#{fixed_fixed_date_string}\"")
          end
        end

        def_node_search :date, <<~EOS
          (pair (sym :date) $str)
        EOS
      end

      # This cop audits deprecate! reason
      class DeprecateDisableReason < FormulaCop
        PUNCTUATION_MARKS = %w[. ! ?].freeze

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          [:deprecate!, :disable!].each do |method|
            node = find_node_method_by_name(body_node, method)

            next if node.nil?

            reason_found = false
            reason(node) do |reason_node|
              reason_found = true
              next if reason_node.sym_type?

              offending_node(reason_node)
              reason_string = string_content(reason_node)

              problem "Do not start the reason with `it`" if reason_string.start_with?("it ")

              problem "Do not end the reason with a punctuation mark" if PUNCTUATION_MARKS.include?(reason_string[-1])
            end

            next if reason_found

            case method
            when :deprecate!
              problem 'Add a reason for deprecation: `deprecate! because: "..."`'
            when :disable!
              problem 'Add a reason for disabling: `disable! because: "..."`'
            end
          end
        end

        def autocorrect(node)
          return unless node.str_type?

          lambda do |corrector|
            reason = string_content(node)
            reason = reason[3..] if reason.start_with?("it ")
            reason.chop! if PUNCTUATION_MARKS.include?(reason[-1])
            corrector.replace(node.source_range, "\"#{reason}\"")
          end
        end

        def_node_search :reason, <<~EOS
          (pair (sym :because) ${str sym})
        EOS
      end
    end
  end
end
