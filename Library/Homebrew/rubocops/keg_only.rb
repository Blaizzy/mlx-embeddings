# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      class KegOnly < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          keg_only_node = find_node_method_by_name(body_node, :keg_only)
          return unless keg_only_node

          whitelist = %w[
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

          reason = string_content(parameters(keg_only_node).first)
          name = Regexp.new(@formula_name, Regexp::IGNORECASE)
          reason = reason.sub(name, "")
          first_word = reason.split.first

          if reason =~ /\A[A-Z]/ && !reason.start_with?(*whitelist)
            problem "'#{first_word}' from the keg_only reason should be '#{first_word.downcase}'."
          end

          return unless reason.end_with?(".")

          problem "keg_only reason should not end with a period."
        end
      end
    end
  end
end
