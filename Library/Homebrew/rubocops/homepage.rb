# typed: true
# frozen_string_literal: true

require "rubocops/extend/formula_cop"
require "rubocops/shared/homepage_helper"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits the `homepage` URL in formulae.
      class Homepage < FormulaCop
        include HomepageHelper
        extend AutoCorrector

        def audit_formula(_node, class_node, _parent_class_node, body_node)
          homepage_node = find_node_method_by_name(body_node, :homepage)

          if homepage_node.nil?
            offending_node(class_node) if body_node.nil?
            problem "Formula should have a homepage."
            return
          end

          homepage_parameter_node = parameters(homepage_node).first
          offending_node(homepage_parameter_node)
          content = string_content(homepage_parameter_node)

          audit_homepage(:formula, content, homepage_node, homepage_parameter_node)
        end
      end
    end
  end
end
