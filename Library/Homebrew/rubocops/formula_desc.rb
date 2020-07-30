# frozen_string_literal: true

require "rubocops/extend/formula"
require "rubocops/shared/desc_helper"
require "extend/string"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits `desc` in Formulae.
      # See the `DescHelper` module for details of the checks.
      class Desc < FormulaCop
        include DescHelper

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          desc_call = find_node_method_by_name(body_node, :desc)
          audit_desc(:formula, @formula_name, desc_call)
        end

        def autocorrect(node)
          autocorrect_desc(node, @formula_name)
        end
      end
    end
  end
end
