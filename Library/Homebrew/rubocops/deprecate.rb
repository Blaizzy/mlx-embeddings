# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits deprecate! date
      class DeprecateDate < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          deprecate_node = find_node_method_by_name(body_node, :deprecate!)

          return if deprecate_node.nil?

          deprecate_date(deprecate_node) do |date_node|
            Date.iso8601(string_content(date_node))
          rescue ArgumentError
            fixed_date_string = Date.parse(string_content(date_node)).iso8601
            offending_node(date_node)
            problem "Use `#{fixed_date_string}` to comply with ISO 8601"
          end
        end

        def autocorrect(node)
          lambda do |corrector|
            fixed_fixed_date_string = Date.parse(string_content(node)).iso8601
            corrector.replace(node.source_range, "\"#{fixed_fixed_date_string}\"")
          end
        end

        def_node_search :deprecate_date, <<~EOS
          (pair (sym :date) $str)
        EOS
      end

      # This cop audits deprecate! reason
      class DeprecateReason < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          deprecate_node = find_node_method_by_name(body_node, :deprecate!)

          return if deprecate_node.nil?

          deprecate_reason(deprecate_node) do |reason_node|
            offending_node(reason_node)
            reason_string = string_content(reason_node)

            problem "Do not start the reason with `it`" if reason_string.start_with?("it ")

            problem "Do not end the reason with a punctuation mark" if %w[. ! ?].include?(reason_string[-1])

            return
          end

          problem 'Add a reason for deprecation: `deprecate! because: "..."`'
        end

        def autocorrect(node)
          return unless node.str_type?

          lambda do |corrector|
            reason = string_content(node)
            reason = reason[3..] if reason.start_with?("it ")
            reason.chop! if %w[. ! ?].include?(reason[-1])
            corrector.replace(node.source_range, "\"#{reason}\"")
          end
        end

        def_node_search :deprecate_reason, <<~EOS
          (pair (sym :because) $str)
        EOS
      end
    end
  end
end
