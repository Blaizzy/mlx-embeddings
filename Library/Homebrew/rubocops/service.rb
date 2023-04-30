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

        CELLAR_PATH_AUDIT_CORRECTIONS = {
          bin:      :opt_bin,
          libexec:  :opt_libexec,
          pkgshare: :opt_pkgshare,
          prefix:   :opt_prefix,
          sbin:     :opt_sbin,
          share:    :opt_share,
        }.freeze

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          service_node = find_block(body_node, :service)
          return if service_node.blank?

          # This check ensures that cellar paths like `bin` are not referenced
          # because their `opt_` variants are more portable and work with the
          # API.
          CELLAR_PATH_AUDIT_CORRECTIONS.each do |path, opt_path|
            find_every_method_call_by_name(service_node, path).each do |node|
              offending_node(node)
              problem "Use `#{opt_path}` instead of `#{path}` in service blocks." do |corrector|
                corrector.replace(node.source_range, opt_path)
              end
            end
          end
        end
      end
    end
  end
end
