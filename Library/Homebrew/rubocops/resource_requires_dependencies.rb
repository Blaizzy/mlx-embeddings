# typed: true
# frozen_string_literal: true

require "rubocops/extend/formula_cop"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits Python formulae that include the "lxml" resource
      # to ensure that they also have the correct `uses_from_macos`
      # dependencies.
      #
      # @api private
      class ResourceRequiresDependencies < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          return if body_node.nil?

          resource_nodes = find_every_method_call_by_name(body_node, :resource)
          lxml = resource_nodes.find { |node| node.arguments.first.str_content == "lxml" }
          return unless lxml

          uses_from_macos_nodes = find_every_method_call_by_name(body_node, :uses_from_macos)
          dependencies = uses_from_macos_nodes.map { |node| node.arguments.first.str_content }
          return if dependencies.include?("libxml2") && dependencies.include?("libxslt")

          offending_node(lxml)
          problem "Add `uses_from_macos` lines above for \"libxml2\"` and \"libxslt\"."
        end
      end
    end
  end
end
