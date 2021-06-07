# frozen_string_literal: true

require 'rubocop'

module RuboCop
  module Cop
    module Sorbet
      # This cop checks that every Ruby file contains a valid Sorbet sigil.
      # Adapted from: https://gist.github.com/clarkdave/85aca4e16f33fd52aceb6a0a29936e52
      #
      # Options:
      #
      # * `RequireSigilOnAllFiles`: make offense if the Sorbet typed is not found in the file (default: false)
      # * `SuggestedStrictness`: Sorbet strictness level suggested in offense messages (default: 'false')
      # * `MinimumStrictness`: If set, make offense if the strictness level in the file is below this one
      #
      # If a `MinimumStrictness` level is specified, it will be used in offense messages and autocorrect.
      class ValidSigil < RuboCop::Cop::Cop
        @registry = Cop.registry # So we can properly subclass this cop

        def investigate(processed_source)
          return if processed_source.tokens.empty?

          sigil = extract_sigil(processed_source)
          return unless check_sigil_present(sigil)

          strictness = extract_strictness(sigil)
          return unless check_strictness_not_empty(sigil, strictness)
          return unless check_strictness_valid(sigil, strictness)
          return unless check_strictness_level(sigil, strictness)
        end

        def autocorrect(_node)
          lambda do |corrector|
            return unless require_sigil_on_all_files?
            return unless extract_sigil(processed_source).nil?

            token = processed_source.tokens.first
            replace_with = suggested_strictness_level(minimum_strictness, suggested_strictness)
            sigil = "# typed: #{replace_with}"
            if token.text.start_with?("#!") # shebang line
              corrector.insert_after(token.pos, "\n#{sigil}")
            else
              corrector.insert_before(token.pos, "#{sigil}\n")
            end
          end
        end

        protected

        STRICTNESS_LEVELS = %w(ignore false true strict strong)
        SIGIL_REGEX = /#\s+typed:(?:\s+([\w]+))?/

        # extraction

        def extract_sigil(processed_source)
          processed_source.tokens
            .take_while { |token| token.type == :tCOMMENT }
            .find { |token| SIGIL_REGEX.match?(token.text) }
        end

        def extract_strictness(sigil)
          sigil.text.match(SIGIL_REGEX)&.captures&.first
        end

        # checks

        def check_sigil_present(sigil)
          return true unless sigil.nil?

          token = processed_source.tokens.first
          if require_sigil_on_all_files?
            strictness = suggested_strictness_level(minimum_strictness, suggested_strictness)
            add_offense(
              token,
              location: token.pos,
              message: 'No Sorbet sigil found in file. ' \
                "Try a `typed: #{strictness}` to start (you can also use `rubocop -a` to automatically add this)."
            )
          end
          false
        end

        def suggested_strictness_level(minimum_strictness, suggested_strictness)
          # if no minimum strictness is set (eg. using Sorbet/HasSigil without config) then
          # we always use the suggested strictness which defaults to `false`
          return suggested_strictness unless minimum_strictness

          # special case: if you're using Sorbet/IgnoreSigil without config, we should recommend `ignore`
          return "ignore" if minimum_strictness == "ignore" && cop_config['SuggestedStrictness'].nil?

          # if a minimum strictness is set (eg. you're using Sorbet/FalseSigil)
          # we want to compare the minimum strictness and suggested strictness. this is because
          # the suggested strictness might be higher than the minimum (eg. if you want all new files
          # at a higher strictness level, without having to migrate existing files at lower levels).

          suggested_level = STRICTNESS_LEVELS.index(suggested_strictness)
          minimum_level = STRICTNESS_LEVELS.index(minimum_strictness)

          suggested_level > minimum_level ? suggested_strictness : minimum_strictness
        end

        def check_strictness_not_empty(sigil, strictness)
          return true if strictness

          add_offense(
            sigil,
            location: sigil.pos,
            message: 'Sorbet sigil should not be empty.'
          )
          false
        end

        def check_strictness_valid(sigil, strictness)
          return true if STRICTNESS_LEVELS.include?(strictness)

          add_offense(
            sigil,
            location: sigil.pos,
            message: "Invalid Sorbet sigil `#{strictness}`."
          )
          false
        end

        def check_strictness_level(sigil, strictness)
          return true unless minimum_strictness

          minimum_level = STRICTNESS_LEVELS.index(minimum_strictness)
          current_level = STRICTNESS_LEVELS.index(strictness)
          if current_level < minimum_level
            add_offense(
              sigil,
              location: sigil.pos,
              message: "Sorbet sigil should be at least `#{minimum_strictness}` got `#{strictness}`."
            )
            return false
          end
          true
        end

        # options

        # Default is `false`
        def require_sigil_on_all_files?
          !!cop_config['RequireSigilOnAllFiles']
        end

        # Default is `'false'`
        def suggested_strictness
          config = cop_config['SuggestedStrictness'].to_s
          STRICTNESS_LEVELS.include?(config) ? config : 'false'
        end

        # Default is `nil`
        def minimum_strictness
          config = cop_config['MinimumStrictness'].to_s
          config if STRICTNESS_LEVELS.include?(config)
        end
      end
    end
  end
end
