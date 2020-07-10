# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      class Text < FormulaCop
        def audit_formula(node, _class_node, _parent_class_node, body_node)
          @full_source_content = source_buffer(node).source

          if match = @full_source_content.match(/^require ['"]formula['"]$/)
            @offensive_node = node
            @source_buf = source_buffer(node)
            @line_no = match.pre_match.count("\n") + 1
            @column = 0
            @length = match[0].length
            @offense_source_range = source_range(@source_buf, @line_no, @column, @length)
            problem "`#{match}` is now unnecessary"
          end

          if !find_node_method_by_name(body_node, :plist_options) &&
             find_method_def(body_node, :plist)
            problem "Please set plist_options when using a formula-defined plist."
          end

          if (depends_on?("openssl") || depends_on?("openssl@1.1")) && depends_on?("libressl")
            problem "Formulae should not depend on both OpenSSL and LibreSSL (even optionally)."
          end

          if formula_tap == "homebrew-core" && (depends_on?("veclibfort") || depends_on?("lapack"))
            problem "Formulae in homebrew/core should use OpenBLAS as the default serial linear algebra library."
          end

          if method_called_ever?(body_node, :virtualenv_create) ||
             method_called_ever?(body_node, :virtualenv_install_with_resources)
            find_method_with_args(body_node, :resource, "setuptools") do
              problem "Formulae using virtualenvs do not need a `setuptools` resource."
            end
          end

          unless method_called_ever?(body_node, :go_resource)
            # processed_source.ast is passed instead of body_node because `require` would be outside body_node
            find_method_with_args(processed_source.ast, :require, "language/go") do
              problem "require \"language/go\" is unnecessary unless using `go_resource`s"
            end
          end

          find_instance_method_call(body_node, "Formula", :factory) do
            problem "\"Formula.factory(name)\" is deprecated in favor of \"Formula[name]\""
          end

          find_every_method_call_by_name(body_node, :xcodebuild).each do |m|
            next if parameters_passed?(m, /SYMROOT=/)

            problem 'xcodebuild should be passed an explicit "SYMROOT"'
          end

          find_method_with_args(body_node, :system, "xcodebuild") do
            problem %q(use "xcodebuild *args" instead of "system 'xcodebuild', *args")
          end

          find_method_with_args(body_node, :system, "go", "get") do
            problem "Do not use `go get`. Please ask upstream to implement Go vendoring"
          end

          find_method_with_args(body_node, :system, "dep", "ensure") do |d|
            next if parameters_passed?(d, /vendor-only/)
            next if @formula_name == "goose" # needed in 2.3.0

            problem "use \"dep\", \"ensure\", \"-vendor-only\""
          end

          find_method_with_args(body_node, :system, "cargo", "build") do
            problem "use \"cargo\", \"install\", *std_cargo_args"
          end

          find_every_method_call_by_name(body_node, :system).each do |m|
            next unless parameters_passed?(m, /make && make/)

            offending_node(m)
            problem "Use separate `make` calls"
          end

          body_node.each_descendant(:dstr) do |dstr_node|
            dstr_node.each_descendant(:begin) do |interpolation_node|
              next unless interpolation_node.source.match?(/#\{\w+\s*\+\s*['"][^}]+\}/)

              offending_node(interpolation_node)
              problem "Do not concatenate paths in string interpolation"
            end
          end

          find_strings(body_node).each do |n|
            next unless regex_match_group(n, /JAVA_HOME/i)

            next if @formula_name.match?(/^openjdk(@|$)/)

            next if find_every_method_call_by_name(body_node, :depends_on).any? do |dependency|
              dependency.each_descendant(:str).count.zero? ||
              regex_match_group(dependency.each_descendant(:str).first, /^openjdk(@|$)/) ||
              depends_on?(:java)
            end

            offending_node(n)
            problem "Use `depends_on :java` to set JAVA_HOME"
          end

          find_strings(body_node).each do |n|
            # Skip strings that don't start with one of the keywords
            next unless regex_match_group(n, %r{^(bin|include|libexec|lib|sbin|share|Frameworks)/?})

            parent = n.parent
            # Only look at keywords that have `prefix` before them
            # TODO: this should be refactored to a direct method match
            prefix_keyword_regex = %r{(prefix\s*\+\s*["'](bin|include|libexec|lib|sbin|share|Frameworks))["'/]}
            if match = parent.source.match(prefix_keyword_regex)
              offending_node(parent)
              problem "Use `#{match[2].downcase}` instead of `#{match[1]}\"`"
            end
          end
        end
      end
    end

    module FormulaAuditStrict
      class Text < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          find_method_with_args(body_node, :go_resource) do
            problem "`go_resource`s are deprecated. Please ask upstream to implement Go vendoring"
          end

          find_method_with_args(body_node, :env, :userpaths) do
            problem "`env :userpaths` in homebrew/core formulae is deprecated"
          end

          body_node.each_descendant(:dstr) do |dstr_node|
            next unless match = dstr_node.source.match(%r{(\#{share}/#{Regexp.escape(@formula_name)})[ /"]})

            offending_node(dstr_node)
            problem "Use `\#{pkgshare}` instead of `#{match[1]}`"
          end

          find_every_method_call_by_name(body_node, :share).each do |share_node|
            if match = share_node.parent.source.match(%r{(share\s*[/+]\s*"#{Regexp.escape(@formula_name)})[/"]})
              offending_node(share_node.parent)
              problem "Use `pkgshare` instead of `#{match[1]}\"`"
            end
          end

          return unless formula_tap == "homebrew-core"

          find_method_with_args(body_node, :env, :std) do
            problem "`env :std` in homebrew/core formulae is deprecated"
          end
        end
      end
    end
  end
end
