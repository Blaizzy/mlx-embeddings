# frozen_string_literal: true

require "rubocops/extend/formula"
require "extend/string"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop audits versioned Formulae for `conflicts_with`.
      class Conflicts < FormulaCop
        MSG = "Versioned formulae should not use `conflicts_with`. " \
              "Use `keg_only :versioned_formula` instead."

        ALLOWLIST = %w[
          bash-completion@2
        ].freeze

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          find_method_calls_by_name(body_node, :conflicts_with).each do |conflicts_with_call|
            next unless parameters(conflicts_with_call).last.respond_to? :values

            reason = parameters(conflicts_with_call).last.values.first
            offending_node(reason)
            name = Regexp.new(@formula_name, Regexp::IGNORECASE)
            reason = string_content(reason).sub(name, "")
            first_word = reason.split.first

            if reason.match?(/\A[A-Z]/)
              problem "'#{first_word}' from the `conflicts_with` reason should be '#{first_word.downcase}'."
            end

            problem "`conflicts_with` reason should not end with a period." if reason.end_with?(".")
          end

          return unless versioned_formula?

          problem MSG if !ALLOWLIST.include?(@formula_name) &&
                         method_called_ever?(body_node, :conflicts_with)
        end

        def autocorrect(node)
          lambda do |corrector|
            if versioned_formula?
              corrector.replace(node.source_range, "keg_only :versioned_formula")
            else
              reason = string_content(node)
              reason[0] = reason[0].downcase
              reason = reason.delete_suffix(".")
              corrector.replace(node.source_range, "\"#{reason}\"")
            end
          end
        end
      end
    end
  end
end
