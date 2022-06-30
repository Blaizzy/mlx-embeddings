# typed: true
# frozen_string_literal: true

require "macos_versions"
require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop checks for various miscellaneous Homebrew coding styles.
      #
      # @api private
      class Lines < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, _body_node)
          [:automake, :ant, :autoconf, :emacs, :expat, :libtool, :mysql, :perl,
           :postgresql, :python, :python3, :rbenv, :ruby].each do |dependency|
            next unless depends_on?(dependency)

            problem ":#{dependency} is deprecated. Usage should be \"#{dependency}\"."
          end

          { apr: "apr-util", fortran: "gcc", gpg: "gnupg", hg: "mercurial",
            mpi: "open-mpi", python2: "python" }.each do |requirement, dependency|
            next unless depends_on?(requirement)

            problem ":#{requirement} is deprecated. Usage should be \"#{dependency}\"."
          end

          problem ":tex is deprecated." if depends_on?(:tex)
        end
      end

      # This cop makes sure that a space is used for class inheritance.
      #
      # @api private
      class ClassInheritance < FormulaCop
        def audit_formula(_node, class_node, parent_class_node, _body_node)
          begin_pos = start_column(parent_class_node)
          end_pos = end_column(class_node)
          return unless begin_pos-end_pos != 3

          problem "Use a space in class inheritance: " \
                  "class #{@formula_name.capitalize} < #{class_name(parent_class_node)}"
        end
      end

      # This cop makes sure that template comments are removed.
      #
      # @api private
      class Comments < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, _body_node)
          audit_comments do |comment|
            [
              "# PLEASE REMOVE",
              "# Documentation:",
              "# if this fails, try separate make/make install steps",
              "# The URL of the archive",
              "## Naming --",
              "# if your formula fails when building in parallel",
              "# Remove unrecognized options if warned by configure",
              '# system "cmake',
            ].each do |template_comment|
              next unless comment.include?(template_comment)

              problem "Please remove default template comments"
              break
            end
          end

          audit_comments do |comment|
            # Commented-out depends_on
            next unless comment =~ /#\s*depends_on\s+(.+)\s*$/

            problem "Commented-out dependency #{Regexp.last_match(1)}"
          end

          return if formula_tap != "homebrew-core"

          # Citation and tag comments from third-party taps
          audit_comments do |comment|
            next if comment !~ /#\s*(cite(?=\s*\w+:)|doi(?=\s*['"])|tag(?=\s*['"]))/

            problem "Formulae in homebrew/core should not use `#{Regexp.last_match(1)}` comments"
          end
        end
      end

      # This cop makes sure that idiomatic `assert_*` statements are used.
      #
      # @api private
      class AssertStatements < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          find_every_method_call_by_name(body_node, :assert).each do |method|
            if method_called_ever?(method, :include?) && !method_called_ever?(method, :!)
              problem "Use `assert_match` instead of `assert ...include?`"
            end

            if method_called_ever?(method, :exist?) && !method_called_ever?(method, :!)
              problem "Use `assert_predicate <path_to_file>, :exist?` instead of `#{method.source}`"
            end

            if method_called_ever?(method, :exist?) && method_called_ever?(method, :!)
              problem "Use `refute_predicate <path_to_file>, :exist?` instead of `#{method.source}`"
            end

            if method_called_ever?(method, :executable?) && !method_called_ever?(method, :!)
              problem "Use `assert_predicate <path_to_file>, :executable?` instead of `#{method.source}`"
            end
          end
        end
      end

      # This cop makes sure that `option`s are used idiomatically.
      #
      # @api private
      class OptionDeclarations < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          problem "Use new-style option definitions" if find_method_def(body_node, :options)

          if formula_tap == "homebrew-core"
            # Use of build.with? implies options, which are forbidden in homebrew/core
            find_instance_method_call(body_node, :build, :without?) do
              problem "Formulae in homebrew/core should not use `build.without?`."
            end
            find_instance_method_call(body_node, :build, :with?) do
              problem "Formulae in homebrew/core should not use `build.with?`."
            end

            return
          end

          depends_on_build_with(body_node) do |build_with_node|
            offending_node(build_with_node)
            problem "Use `:optional` or `:recommended` instead of `if #{build_with_node.source}`"
          end

          find_instance_method_call(body_node, :build, :without?) do |method|
            next unless unless_modifier?(method.parent)

            correct = method.source.gsub("out?", "?")
            problem "Use if #{correct} instead of unless #{method.source}"
          end

          find_instance_method_call(body_node, :build, :with?) do |method|
            next unless unless_modifier?(method.parent)

            correct = method.source.gsub("?", "out?")
            problem "Use if #{correct} instead of unless #{method.source}"
          end

          find_instance_method_call(body_node, :build, :with?) do |method|
            next unless expression_negated?(method)

            problem "Don't negate 'build.with?': use 'build.without?'"
          end

          find_instance_method_call(body_node, :build, :without?) do |method|
            next unless expression_negated?(method)

            problem "Don't negate 'build.without?': use 'build.with?'"
          end

          find_instance_method_call(body_node, :build, :without?) do |method|
            arg = parameters(method).first
            next unless (match = regex_match_group(arg, /^-?-?without-(.*)/))

            problem "Don't duplicate 'without': " \
                    "Use `build.without? \"#{match[1]}\"` to check for \"--without-#{match[1]}\""
          end

          find_instance_method_call(body_node, :build, :with?) do |method|
            arg = parameters(method).first
            next unless (match = regex_match_group(arg, /^-?-?with-(.*)/))

            problem "Don't duplicate 'with': Use `build.with? \"#{match[1]}\"` to check for \"--with-#{match[1]}\""
          end

          find_instance_method_call(body_node, :build, :include?) do
            problem "`build.include?` is deprecated"
          end
        end

        def unless_modifier?(node)
          return false unless node.if_type?

          node.modifier_form? && node.unless?
        end

        # Finds `depends_on "foo" if build.with?("bar")` or `depends_on "foo" if build.without?("bar")`
        def_node_search :depends_on_build_with, <<~EOS
          (if $(send (send nil? :build) {:with? :without?} str)
            (send nil? :depends_on str) nil?)
        EOS
      end

      # This cop makes sure that formulae depend on `open-mpi` instead of `mpich`.
      #
      # @api private
      class MpiCheck < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          # Enforce use of OpenMPI for MPI dependency in core
          return unless formula_tap == "homebrew-core"

          find_method_with_args(body_node, :depends_on, "mpich") do
            problem "Formulae in homebrew/core should use 'depends_on \"open-mpi\"' " \
                    "instead of '#{@offensive_node.source}'." do |corrector|
              corrector.replace(@offensive_node.source_range, "depends_on \"open-mpi\"")
            end
          end
        end
      end

      # This cop makes sure that formulae do not depend on `pyoxidizer` at build-time
      # or run-time.
      #
      # @api private
      class PyoxidizerCheck < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          # Disallow use of PyOxidizer as a dependency in core
          return unless formula_tap == "homebrew-core"

          find_method_with_args(body_node, :depends_on, "pyoxidizer") do
            problem "Formulae in homebrew/core should not use '#{@offensive_node.source}'."
          end

          [
            :build,
            [:build],
            [:build, :test],
            [:test, :build],
          ].each do |type|
            find_method_with_args(body_node, :depends_on, "pyoxidizer" => type) do
              problem "Formulae in homebrew/core should not use '#{@offensive_node.source}'."
            end
          end
        end
      end

      # This cop makes sure that the safe versions of `popen_*` calls are used.
      #
      # @api private
      class SafePopenCommands < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          test = find_block(body_node, :test)

          [:popen_read, :popen_write].each do |unsafe_command|
            test_methods = []

            unless test.nil?
              find_instance_method_call(test, "Utils", unsafe_command) do |method|
                test_methods << method.source_range
              end
            end

            find_instance_method_call(body_node, "Utils", unsafe_command) do |method|
              unless test_methods.include?(method.source_range)
                problem "Use `Utils.safe_#{unsafe_command}` instead of `Utils.#{unsafe_command}`" do |corrector|
                  corrector.replace(@offensive_node.loc.selector, "safe_#{@offensive_node.method_name}")
                end
              end
            end
          end
        end
      end

      # This cop makes sure that environment variables are passed correctly to `popen_*` calls.
      #
      # @api private
      class ShellVariables < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          popen_commands = [
            :popen,
            :popen_read,
            :safe_popen_read,
            :popen_write,
            :safe_popen_write,
          ]

          popen_commands.each do |command|
            find_instance_method_call(body_node, "Utils", command) do |method|
              next unless (match = regex_match_group(parameters(method).first, /^([^"' ]+)=([^"' ]+)(?: (.*))?$/))

              good_args = "Utils.#{command}({ \"#{match[1]}\" => \"#{match[2]}\" }, \"#{match[3]}\")"

              problem "Use `#{good_args}` instead of `#{method.source}`" do |corrector|
                corrector.replace(@offensive_node.source_range,
                                  "{ \"#{match[1]}\" => \"#{match[2]}\" }, \"#{match[3]}\"")
              end
            end
          end
        end
      end

      # This cop makes sure that `license` has the correct format.
      #
      # @api private
      class LicenseArrays < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          license_node = find_node_method_by_name(body_node, :license)
          return unless license_node

          license = parameters(license_node).first
          return unless license.array_type?

          problem "Use `license any_of: #{license.source}` instead of `license #{license.source}`" do |corrector|
            corrector.replace(license_node.source_range, "license any_of: #{parameters(license_node).first.source}")
          end
        end
      end

      # This cop makes sure that nested `license` declarations are split onto multiple lines.
      #
      # @api private
      class Licenses < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          license_node = find_node_method_by_name(body_node, :license)
          return unless license_node
          return if license_node.source.include?("\n")

          parameters(license_node).first.each_descendant(:hash).each do |license_hash|
            next if license_exception? license_hash

            problem "Split nested license declarations onto multiple lines"
          end
        end

        def_node_matcher :license_exception?, <<~EOS
          (hash (pair (sym :with) str))
        EOS
      end

      # This cop makes sure that Python versions are consistent.
      #
      # @api private
      class PythonVersions < FormulaCop
        extend AutoCorrector

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          python_formula_node = find_every_method_call_by_name(body_node, :depends_on).find do |dep|
            string_content(parameters(dep).first).start_with? "python@"
          end

          return if python_formula_node.blank?

          python_version = string_content(parameters(python_formula_node).first).split("@").last

          find_strings(body_node).each do |str|
            content = string_content(str)

            next unless (match = content.match(/^python(@)?(\d\.\d+)$/))
            next if python_version == match[2]

            fix = if match[1]
              "python@#{python_version}"
            else
              "python#{python_version}"
            end

            offending_node(str)
            problem "References to `#{content}` should " \
                    "match the specified python dependency (`#{fix}`)" do |corrector|
              corrector.replace(str.source_range, "\"#{fix}\"")
            end
          end
        end
      end

      # This cop makes sure that OS conditionals are consistent.
      #
      # @api private
      class OnSystemConditionals < FormulaCop
        extend AutoCorrector

        NO_ON_SYSTEM_METHOD_NAMES = [:install, :post_install].freeze
        NO_ON_SYSTEM_BLOCK_NAMES = [:service, :test].freeze

        MACOS_VERSION_CONDITIONALS = {
          "==" => nil,
          "<=" => :or_older,
          ">=" => :or_newer,
        }.freeze

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          on_arch_options = [:intel, :arm]
          on_base_os_options = [:macos, :linux]
          on_macos_version_options = MacOSVersions::SYMBOLS.keys
          all_system_options = on_arch_options + on_base_os_options + on_macos_version_options

          top_level_nodes_to_check = []
          NO_ON_SYSTEM_METHOD_NAMES.each do |formula_method_name|
            method_node = find_method_def(body_node, formula_method_name)
            top_level_nodes_to_check << [formula_method_name, method_node] if method_node
          end
          NO_ON_SYSTEM_BLOCK_NAMES.each do |formula_block_name|
            block_node = find_block(body_node, formula_block_name)
            top_level_nodes_to_check << [formula_block_name, block_node] if block_node
          end

          all_system_options.each do |on_system_option|
            on_system_method = :"on_#{on_system_option}"
            if_module_name, if_method_name = if on_arch_options.include?(on_system_option)
              ["Hardware::CPU", :"#{on_system_option}?"]
            elsif on_base_os_options.include?(on_system_option)
              ["OS", on_system_option == :macos ? :mac? : :linux?]
            else
              ["MacOS", :version]
            end
            on_system_method_string = on_system_method.to_s
            if_statement_string = "if #{if_module_name}.#{if_method_name}"

            top_level_nodes_to_check.each do |top_level_name, top_level_node|
              top_level_node_string = if top_level_node.def_type?
                "def #{top_level_name}"
              else
                "#{top_level_name} do"
              end

              find_every_method_call_by_name(top_level_node, on_system_method).each do |method|
                if on_macos_version_options.include?(on_system_option)
                  on_macos_version_method_call(method, on_method: on_system_method) do |on_method_parameters|
                    if on_method_parameters.empty?
                      if_statement_string = "#{if_statement_string} == :#{on_system_option}"
                    else
                      on_system_method_string = "#{on_system_method} :#{on_method_parameters.first}"
                      if_condition_operator = MACOS_VERSION_CONDITIONALS.key(on_method_parameters.first)
                      if_statement_string = "#{if_statement_string} #{if_condition_operator} :#{on_system_option}"
                    end
                  end
                end

                offending_node(method)

                problem "Don't use `#{on_system_method_string}` in `#{top_level_node_string}`, " \
                        "use `#{if_statement_string}` instead." do |corrector|
                  block_node = offending_node.parent
                  next if block_node.type != :block

                  # TODO: could fix corrector to handle this but punting for now.
                  next if block_node.single_line?

                  source_range = offending_node.source_range.join(offending_node.parent.loc.begin)
                  corrector.replace(source_range, if_statement_string)
                end
              end
            end

            # Don't restrict OS.mac? or OS.linux? usage in taps; they don't care
            # as much as we do about e.g. formulae.brew.sh generation, often use
            # platform-specific URLs and we don't want to add DSLs to support
            # that case.
            next if formula_tap != "homebrew-core"

            if_nodes_to_check = []

            if on_arch_options.include?(on_system_option)
              if_arch_node_search(body_node, arch: if_method_name) do |if_node, else_node|
                else_info = if else_node.present?
                  {
                    can_autocorrect:  true,
                    on_system_method: on_system_option == :intel ? "on_arm" : "on_intel",
                    node:             else_node,
                  }
                end

                if_nodes_to_check << [if_node, else_info]
              end
            elsif on_base_os_options.include?(on_system_option)
              if_base_os_node_search(body_node, base_os: if_method_name) do |if_node, else_node|
                else_info = if else_node.present?
                  {
                    can_autocorrect:  true,
                    on_system_method: on_system_option == :macos ? "on_linux" : "on_macos",
                    node:             else_node,
                  }
                end

                if_nodes_to_check << [if_node, else_info]
              end
            else
              if_macos_version_node_search(body_node, os_name: on_system_option) do |if_node, operator, else_node|
                if operator != :== && MACOS_VERSION_CONDITIONALS.key?(operator.to_s)
                  on_system_method_string = "#{on_system_method} :#{MACOS_VERSION_CONDITIONALS[operator.to_s]}"
                end
                if_statement_string = "if #{if_module_name}.#{if_method_name} #{operator} :#{on_system_option}"
                if else_node.present? || !MACOS_VERSION_CONDITIONALS.key?(operator.to_s)
                  else_info = { can_autocorrect: false }
                end

                if_nodes_to_check << [if_node, else_info]
              end
            end

            if_nodes_to_check.each do |if_node, else_info|
              # TODO: check to see if it's legal
              valid = T.let(false, T::Boolean)
              if_node.each_ancestor do |ancestor|
                valid_method_names = case ancestor.type
                when :def
                  NO_ON_SYSTEM_METHOD_NAMES
                when :block
                  NO_ON_SYSTEM_BLOCK_NAMES
                else
                  next
                end
                next unless valid_method_names.include?(ancestor.method_name)

                valid = true
                break
              end
              next if valid

              offending_node(if_node)

              problem "Don't use `#{if_statement_string}`, " \
                      "use `#{on_system_method_string} do` instead." do |corrector|
                # TODO: could fix corrector to handle this but punting for now.
                next if if_node.unless?

                if else_info.present?
                  next unless else_info[:can_autocorrect]

                  corrector.replace(if_node.source_range,
                                    "#{on_system_method_string} do\n#{if_node.body.source}\nend\n" \
                                    "#{else_info[:on_system_method]} do\n#{else_info[:node].source}\nend")
                else
                  corrector.replace(if_node.source_range,
                                    "#{on_system_method_string} do\n#{if_node.body.source}\nend")
                end
              end
            end
          end
        end

        def_node_matcher :on_macos_version_method_call, <<~PATTERN
          (send nil? %on_method (sym ${:or_newer :or_older})?)
        PATTERN

        def_node_search :if_arch_node_search, <<~PATTERN
          $(if (send (const (const nil? :Hardware) :CPU) %arch) _ $_)
        PATTERN

        def_node_search :if_base_os_node_search, <<~PATTERN
          $(if (send (const nil? :OS) %base_os) _ $_)
        PATTERN

        def_node_search :if_macos_version_node_search, <<~PATTERN
          $(if (send (send (const nil? :MacOS) :version) ${:== :<= :< :>= :> :!=} (sym %os_name)) _ $_)
        PATTERN
      end

      # This cop checks for other miscellaneous style violations.
      #
      # @api private
      class Miscellaneous < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          # FileUtils is included in Formula
          # encfs modifies a file with this name, so check for some leading characters
          find_instance_method_call(body_node, "FileUtils", nil) do |method_node|
            problem "Don't need 'FileUtils.' before #{method_node.method_name}"
          end

          # Check for long inreplace block vars
          find_all_blocks(body_node, :inreplace) do |node|
            block_arg = node.arguments.children.first
            next unless block_arg.source.size > 1

            problem "\"inreplace <filenames> do |s|\" is preferred over \"|#{block_arg.source}|\"."
          end

          [:rebuild, :version_scheme].each do |method_name|
            find_method_with_args(body_node, method_name, 0) do
              problem "'#{method_name} 0' should be removed"
            end
          end

          find_instance_call(body_node, "ARGV") do |_method_node|
            problem "Use build instead of ARGV to check options"
          end

          find_instance_method_call(body_node, :man, :+) do |method|
            next unless (match = regex_match_group(parameters(method).first, /^man[1-8]$/))

            problem "\"#{method.source}\" should be \"#{match[0]}\""
          end

          # Avoid hard-coding compilers
          find_every_method_call_by_name(body_node, :system).each do |method|
            param = parameters(method).first
            if (match = regex_match_group(param, %r{^(/usr/bin/)?(gcc|llvm-gcc|clang)(\s|$)}))
              problem "Use \"\#{ENV.cc}\" instead of hard-coding \"#{match[2]}\""
            elsif (match = regex_match_group(param, %r{^(/usr/bin/)?((g|llvm-g|clang)\+\+)(\s|$)}))
              problem "Use \"\#{ENV.cxx}\" instead of hard-coding \"#{match[2]}\""
            end
          end

          find_instance_method_call(body_node, "ENV", :[]=) do |method|
            param = parameters(method)[1]
            if (match = regex_match_group(param, %r{^(/usr/bin/)?(gcc|llvm-gcc|clang)(\s|$)}))
              problem "Use \"\#{ENV.cc}\" instead of hard-coding \"#{match[2]}\""
            elsif (match = regex_match_group(param, %r{^(/usr/bin/)?((g|llvm-g|clang)\+\+)(\s|$)}))
              problem "Use \"\#{ENV.cxx}\" instead of hard-coding \"#{match[2]}\""
            end
          end

          # Prefer formula path shortcuts in strings
          formula_path_strings(body_node, :share) do |p|
            next unless (match = regex_match_group(p, %r{^(/(man))/?}))

            problem "\"\#{share}#{match[1]}\" should be \"\#{#{match[2]}}\""
          end

          formula_path_strings(body_node, :prefix) do |p|
            if (match = regex_match_group(p, %r{^(/share/(info|man))$}))
              problem "\"\#\{prefix}#{match[1]}\" should be \"\#{#{match[2]}}\""
            end
            if (match = regex_match_group(p, %r{^((/share/man/)(man[1-8]))}))
              problem "\"\#\{prefix}#{match[1]}\" should be \"\#{#{match[3]}}\""
            end
            if (match = regex_match_group(p, %r{^(/(bin|include|libexec|lib|sbin|share|Frameworks))}i))
              problem "\"\#\{prefix}#{match[1]}\" should be \"\#{#{match[2].downcase}}\""
            end
          end

          find_every_method_call_by_name(body_node, :depends_on).each do |method|
            key, value = destructure_hash(parameters(method).first)
            next if key.nil? || value.nil?
            next unless (match = regex_match_group(value, /^(lua|perl|python|ruby)(\d*)/))

            problem "#{match[1]} modules should be vendored rather than use deprecated `#{method.source}`"
          end

          find_every_method_call_by_name(body_node, :system).each do |method|
            next unless (match = regex_match_group(parameters(method).first, /^(env|export)(\s+)?/))

            problem "Use ENV instead of invoking '#{match[1]}' to modify the environment"
          end

          find_every_method_call_by_name(body_node, :depends_on).each do |method|
            param = parameters(method).first
            dep, option_child_nodes = hash_dep(param)
            next if dep.nil? || option_child_nodes.empty?

            option_child_nodes.each do |option|
              find_strings(option).each do |dependency|
                next unless (match = regex_match_group(dependency, /(with(out)?-\w+|c\+\+11)/))

                problem "Dependency #{string_content(dep)} should not use option #{match[0]}"
              end
            end
          end

          find_instance_method_call(body_node, :version, :==) do |method|
            next unless parameters_passed?(method, "HEAD")

            problem "Use 'build.head?' instead of inspecting 'version'"
          end

          find_instance_method_call(body_node, "ARGV", :include?) do |method|
            next unless parameters_passed?(method, "--HEAD")

            problem "Use \"if build.head?\" instead"
          end

          find_const(body_node, "MACOS_VERSION") do
            problem "Use MacOS.version instead of MACOS_VERSION"
          end

          find_const(body_node, "MACOS_FULL_VERSION") do
            problem "Use MacOS.full_version instead of MACOS_FULL_VERSION"
          end

          conditional_dependencies(body_node) do |node, method, param, dep_node|
            dep = string_content(dep_node)
            if node.if?
              if (method == :include? && regex_match_group(param, /^with-#{dep}$/)) ||
                 (method == :with? && regex_match_group(param, /^#{dep}$/))
                offending_node(dep_node.parent)
                problem "Replace #{node.source} with #{dep_node.parent.source} => :optional"
              end
            elsif node.unless?
              if (method == :include? && regex_match_group(param, /^without-#{dep}$/)) ||
                 (method == :without? && regex_match_group(param, /^#{dep}$/))
                offending_node(dep_node.parent)
                problem "Replace #{node.source} with #{dep_node.parent.source} => :recommended"
              end
            end
          end

          find_method_with_args(body_node, :fails_with, :llvm) do
            problem "'fails_with :llvm' is now a no-op so should be removed"
          end

          find_method_with_args(body_node, :needs, :openmp) do
            problem "'needs :openmp' should be replaced with 'depends_on \"gcc\"'"
          end

          find_method_with_args(body_node, :system, /^(otool|install_name_tool|lipo)/) do
            problem "Use ruby-macho instead of calling #{@offensive_node.source}"
          end

          find_every_method_call_by_name(body_node, :system).each do |method_node|
            # Skip Kibana: npm cache edge (see formula for more details)
            next if @formula_name.match?(/^kibana(@\d[\d.]*)?$/)

            first_param, second_param = parameters(method_node)
            next if !node_equals?(first_param, "npm") ||
                    !node_equals?(second_param, "install")

            offending_node(method_node)
            problem "Use Language::Node for npm install args" unless languageNodeModule?(method_node)
          end

          problem "Use new-style test definitions (test do)" if find_method_def(body_node, :test)

          find_method_with_args(body_node, :skip_clean, :all) do
            problem "`skip_clean :all` is deprecated; brew no longer strips symbols. " \
                    "Pass explicit paths to prevent Homebrew from removing empty folders."
          end

          if find_method_def(processed_source.ast)
            problem "Define method #{method_name(@offensive_node)} in the class body, not at the top-level"
          end

          find_instance_method_call(body_node, :build, :universal?) do
            next if @formula_name == "wine"

            problem "macOS has been 64-bit only since 10.6 so build.universal? is deprecated."
          end

          find_instance_method_call(body_node, "ENV", :universal_binary) do
            next if @formula_name == "wine"

            problem "macOS has been 64-bit only since 10.6 so ENV.universal_binary is deprecated."
          end

          find_instance_method_call(body_node, "ENV", :runtime_cpu_detection) do
            next if tap_style_exception? :runtime_cpu_detection_allowlist

            problem "Formulae should be verified as having support for runtime hardware detection before " \
                    "using ENV.runtime_cpu_detection."
          end

          find_every_method_call_by_name(body_node, :depends_on).each do |method|
            next unless method_called?(method, :new)

            problem "`depends_on` can take requirement classes instead of instances"
          end

          find_instance_method_call(body_node, "ENV", :[]) do |method|
            next unless modifier?(method.parent)

            param = parameters(method).first
            next unless node_equals?(param, "CI")

            problem 'Don\'t use ENV["CI"] for Homebrew CI checks.'
          end

          find_instance_method_call(body_node, "Dir", :[]) do |method|
            next unless parameters(method).size == 1

            path = parameters(method).first
            next unless path.str_type?
            next unless (match = regex_match_group(path, /^[^*{},]+$/))

            problem "Dir([\"#{string_content(path)}\"]) is unnecessary; just use \"#{match[0]}\""
          end

          fileutils_methods = Regexp.new(
            FileUtils.singleton_methods(false)
                     .map { |m| "(?-mix:^#{Regexp.escape(m)}$)" }
                     .join("|"),
          )
          find_every_method_call_by_name(body_node, :system).each do |method|
            param = parameters(method).first
            next unless (match = regex_match_group(param, fileutils_methods))

            problem "Use the `#{match}` Ruby method instead of `#{method.source}`"
          end
        end

        def modifier?(node)
          return false unless node.if_type?

          node.modifier_form?
        end

        def_node_search :conditional_dependencies, <<~EOS
          {$(if (send (send nil? :build) ${:include? :with? :without?} $(str _))
              (send nil? :depends_on $({str sym} _)) nil?)

           $(if (send (send nil? :build) ${:include? :with? :without?} $(str _)) nil?
              (send nil? :depends_on $({str sym} _)))}
        EOS

        def_node_matcher :hash_dep, <<~EOS
          (hash (pair $(str _) $...))
        EOS

        def_node_matcher :destructure_hash, <<~EOS
          (hash (pair $(str _) $(sym _)))
        EOS

        def_node_search :formula_path_strings, <<~EOS
          {(dstr (begin (send nil? %1)) $(str _ ))
           (dstr _ (begin (send nil? %1)) $(str _ ))}
        EOS

        # Node Pattern search for Language::Node
        def_node_search :languageNodeModule?, <<~EOS
          (const (const nil? :Language) :Node)
        EOS
      end
    end

    module FormulaAuditStrict
      # This cop makes sure that no build-time checks are performed.
      #
      # @api private
      class MakeCheck < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          return if formula_tap != "homebrew-core"

          # Avoid build-time checks in homebrew/core
          find_every_method_call_by_name(body_node, :system).each do |method|
            next if @formula_name.start_with?("lib")
            next if tap_style_exception? :make_check_allowlist

            params = parameters(method)
            next unless node_equals?(params[0], "make")

            params[1..].each do |arg|
              next unless regex_match_group(arg, /^(checks?|tests?)$/)

              offending_node(method)
              problem "Formulae in homebrew/core (except e.g. cryptography, libraries) " \
                      "should not run build-time checks"
            end
          end
        end
      end

      # This cop ensures that new formulae depending on removed Requirements are not used
      class Requirements < FormulaCop
        def audit_formula(_node, _class_node, _parent_class_node, _body_node)
          problem "Formulae should depend on a versioned `openjdk` instead of :java" if depends_on? :java
          problem "Formulae should depend on specific X libraries instead of :x11" if depends_on? :x11
          problem "Formulae should not depend on :osxfuse" if depends_on? :osxfuse
          problem "Formulae should not depend on :tuntap" if depends_on? :tuntap
        end
      end
    end
  end
end
