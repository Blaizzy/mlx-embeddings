# typed: false
# frozen_string_literal: true

require "ast_constants"
require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop checks for correct order of components in formulae.
      #
      # - `component_precedence_list` has component hierarchy in a nested list
      #   where each sub array contains components' details which are at same precedence level
      class ComponentsOrder < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          @present_components, @offensive_nodes = check_order(FORMULA_COMPONENT_PRECEDENCE_LIST, body_node)

          component_problem @offensive_nodes[0], @offensive_nodes[1] if @offensive_nodes

          component_precedence_list = [
            [{ name: :depends_on, type: :method_call }],
            [{ name: :resource, type: :block_call }],
            [{ name: :patch, type: :method_call }, { name: :patch, type: :block_call }],
          ]

          on_macos_blocks = find_blocks(body_node, :on_macos)

          if on_macos_blocks.length > 1
            @offensive_node = on_macos_blocks.second
            problem "there can only be one `on_macos` block in a formula."
          end

          check_on_os_block_content(component_precedence_list, on_macos_blocks.first) if on_macos_blocks.any?

          on_linux_blocks = find_blocks(body_node, :on_linux)

          if on_linux_blocks.length > 1
            @offensive_node = on_linux_blocks.second
            problem "there can only be one `on_linux` block in a formula."
          end

          check_on_os_block_content(component_precedence_list, on_linux_blocks.first) if on_linux_blocks.any?

          resource_blocks = find_blocks(body_node, :resource)
          resource_blocks.each do |resource_block|
            on_macos_blocks = find_blocks(resource_block.body, :on_macos)
            on_linux_blocks = find_blocks(resource_block.body, :on_linux)

            if on_macos_blocks.length.zero? && on_linux_blocks.length.zero?
              # Found nothing. Try without .body as depending on the code,
              # on_macos or on_linux might be in .body or not ...
              on_macos_blocks = find_blocks(resource_block, :on_macos)
              on_linux_blocks = find_blocks(resource_block, :on_linux)

              next if on_macos_blocks.length.zero? && on_linux_blocks.length.zero?
            end

            @offensive_node = resource_block

            next if on_macos_blocks.length.zero? && on_linux_blocks.length.zero?

            on_os_bodies = []

            (on_macos_blocks + on_linux_blocks).each do |on_os_block|
              on_os_body = on_os_block.body
              branches = on_os_body.if_type? ? on_os_body.branches : [on_os_body]
              on_os_bodies += branches.map { |branch| [on_os_block, branch] }
            end

            message = nil
            allowed_methods = [
              [:url, :sha256],
              [:url, :mirror, :sha256],
              [:url, :version, :sha256],
              [:url, :mirror, :version, :sha256],
            ]
            minimum_methods = allowed_methods.first.map { |m| "`#{m}`" }.to_sentence
            maximum_methods = allowed_methods.last.map { |m| "`#{m}`" }.to_sentence

            on_os_bodies.each do |on_os_block, on_os_body|
              method_name = on_os_block.method_name
              child_nodes = on_os_body.begin_type? ? on_os_body.child_nodes : [on_os_body]
              if child_nodes.all? { |n| n.send_type? || n.block_type? }
                method_names = child_nodes.map(&:method_name)
                next if allowed_methods.include? method_names
              end
              offending_node(on_os_block)
              message = "`#{method_name}` blocks within `resource` blocks must contain at least "\
                        "#{minimum_methods} and at most #{maximum_methods} (in order)."
              break
            end

            if message.present?
              problem message
              next
            end

            if on_macos_blocks.length > 1
              problem "there can only be one `on_macos` block in a resource block."
              next
            end

            if on_linux_blocks.length > 1
              problem "there can only be one `on_linux` block in a resource block."
              next
            end
          end
        end

        def check_on_os_block_content(component_precedence_list, on_os_block)
          on_os_allowed_methods = %w[depends_on patch resource deprecate! disable!]
          _, offensive_node = check_order(component_precedence_list, on_os_block.body)
          component_problem(*offensive_node) if offensive_node
          child_nodes = on_os_block.body.begin_type? ? on_os_block.body.child_nodes : [on_os_block.body]
          child_nodes.each do |child|
            valid_node = depends_on_node?(child)
            # Check for RuboCop::AST::SendNode and RuboCop::AST::BlockNode instances
            # only, as we are checking the method_name for `patch`, `resource`, etc.
            method_type = child.send_type? || child.block_type?
            next unless method_type

            valid_node ||= on_os_allowed_methods.include? child.method_name.to_s

            @offensive_node = child
            next if valid_node

            problem "`#{on_os_block.method_name}` cannot include `#{child.method_name}`. " \
                    "Only #{on_os_allowed_methods.map { |m| "`#{m}`" }.to_sentence} are allowed."
          end
        end

        # Reorder two nodes in the source, using the corrector instance in autocorrect method.
        # Components of same type are grouped together when rewriting the source.
        # Linebreaks are introduced if components are of two different methods/blocks/multilines.
        def reorder_components(corrector, node1, node2)
          # order_idx : node1's index in component_precedence_list
          # curr_p_idx: node1's index in preceding_comp_arr
          # preceding_comp_arr: array containing components of same type
          order_idx, curr_p_idx, preceding_comp_arr = get_state(node1)

          # curr_p_idx.positive? means node1 needs to be grouped with its own kind
          if curr_p_idx.positive?
            node2 = preceding_comp_arr[curr_p_idx - 1]
            indentation = " " * (start_column(node2) - line_start_column(node2))
            line_breaks = node2.multiline? ? "\n\n" : "\n"
            corrector.insert_after(node2.source_range, line_breaks + indentation + node1.source)
          else
            indentation = " " * (start_column(node2) - line_start_column(node2))
            # No line breaks up to version_scheme, order_idx == 8
            line_breaks = (order_idx > 8) ? "\n\n" : "\n"
            corrector.insert_before(node2.source_range, node1.source + line_breaks + indentation)
          end
          corrector.remove(range_with_surrounding_space(range: node1.source_range, side: :left))
        end

        # Returns precedence index and component's index to properly reorder and group during autocorrect.
        def get_state(node1)
          @present_components.each_with_index do |comp, idx|
            return [idx, comp.index(node1), comp] if comp.member?(node1)
          end
        end

        def check_order(component_precedence_list, body_node)
          present_components = component_precedence_list.map do |components|
            components.flat_map do |component|
              case component[:type]
              when :method_call
                find_method_calls_by_name(body_node, component[:name]).to_a
              when :block_call
                find_blocks(body_node, component[:name]).to_a
              when :method_definition
                find_method_def(body_node, component[:name])
              end
            end.compact
          end

          # Check if each present_components is above rest of the present_components
          offensive_nodes = nil
          present_components.take(present_components.size - 1).each_with_index do |preceding_component, p_idx|
            next if preceding_component.empty?

            present_components.drop(p_idx + 1).each do |succeeding_component|
              next if succeeding_component.empty?

              offensive_nodes = check_precedence(preceding_component, succeeding_component)
              return [present_components, offensive_nodes] if offensive_nodes
            end
          end
          nil
        end

        # Method to report and correct component precedence violations.
        def component_problem(c1, c2)
          return if tap_style_exception? :components_order_exceptions

          problem "`#{format_component(c1)}` (line #{line_number(c1)}) " \
            "should be put before `#{format_component(c2)}` " \
            "(line #{line_number(c2)})" do |corrector|
            reorder_components(corrector, c1, c2)
          end
        end

        # Node pattern method to match
        # `depends_on` variants.
        def_node_matcher :depends_on_node?, <<~EOS
          {(if _ (send nil? :depends_on ...) nil?)
           (send nil? :depends_on ...)}
        EOS
      end
    end
  end
end
