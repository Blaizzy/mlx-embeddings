# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      class KegOnly < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          keg_only_node = find_node_method_by_name(body_node, :keg_only)
          return unless keg_only_node

          allowlist = %w[
            Apple
            macOS
            OS
            Homebrew
            Xcode
            GPG
            GNOME
            BSD
            Firefox
          ].freeze

          reason = parameters(keg_only_node).first
          offending_node(reason)
          name = Regexp.new(@formula_name, Regexp::IGNORECASE)
          reason = string_content(reason).sub(name, "")
          first_word = reason.split.first

          if reason =~ /\A[A-Z]/ && !reason.start_with?(*allowlist)
            problem "'#{first_word}' from the `keg_only` reason should be '#{first_word.downcase}'."
          end

          return unless reason.end_with?(".")

          problem "`keg_only` reason should not end with a period."
        end

        def autocorrect(node)
          lambda do |corrector|
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
