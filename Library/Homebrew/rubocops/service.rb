# typed: true
# frozen_string_literal: true

require "rubocops/extend/formula_cop"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits the service block.
      #
      # @api private
      class Service < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          service_node = find_block(body_node, :service)
          return if service_node.blank?

          # This check ensures that `bin` is not referenced because
          # `opt_bin` is more portable and works with the API.
          find_every_method_call_by_name(service_node, :bin).each do |bin_node|
            offending_node(bin_node)
            problem "Use `opt_bin` instead of `bin` in service blocks." do |corrector|
              corrector.replace(bin_node.source_range, "opt_bin")
            end
          end
        end
      end
    end
  end
end
