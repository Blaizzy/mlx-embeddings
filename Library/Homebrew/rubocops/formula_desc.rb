# typed: true
# frozen_string_literal: true

require "rubocops/extend/formula"
require "rubocops/shared/desc_helper"
require "extend/string"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits `desc` in formulae.
      # See the {DescHelper} module for details of the checks.
      class Desc < FormulaCop
        include DescHelper
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          @name = @formula_name
          desc_call = find_node_method_by_name(body_node, :desc)
          audit_desc(:formula, @name, desc_call)
        end
      end
    end
  end
end
