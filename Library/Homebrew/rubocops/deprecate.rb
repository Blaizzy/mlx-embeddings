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
    end
  end
end
