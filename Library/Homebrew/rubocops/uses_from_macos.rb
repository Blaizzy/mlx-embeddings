# typed: true
# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits formulae that are keg-only because they are provided by macos.
      class ProvidedByMacos < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          find_method_with_args(body_node, :keg_only, :provided_by_macos) do
            unless tap_style_exception? :provided_by_macos_formulae
              problem "Formulae that are `keg_only :provided_by_macos` should be added to "\
                      "`style_exceptions/provided_by_macos_formulae.json`"
            end
          end
        end
      end

      # This cop audits `uses_from_macos` dependencies in formulae.
      class UsesFromMacos < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          find_method_with_args(body_node, :uses_from_macos, /^"(.+)"/).each do |method|
            dep = if parameters(method).first.instance_of?(RuboCop::AST::StrNode)
              parameters(method).first
            elsif parameters(method).first.instance_of?(RuboCop::AST::HashNode)
              parameters(method).first.keys.first
            end

            next if tap_style_exception? :provided_by_macos_formulae, string_content(dep)
            next if tap_style_exception? :non_keg_only_provided_by_macos_formulae, string_content(dep)

            problem "`uses_from_macos` should only be used for macOS dependencies, not #{string_content(dep)}."
          end
        end
      end
    end
  end
end
