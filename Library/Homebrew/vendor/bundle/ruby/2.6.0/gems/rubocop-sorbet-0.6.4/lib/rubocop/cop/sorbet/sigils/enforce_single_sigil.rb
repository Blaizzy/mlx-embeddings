# frozen_string_literal: true

module RuboCop
  module Cop
    module Sorbet
      # This cop checks that there is only one Sorbet sigil in a given file
      #
      # For example, the following class with two sigils
      #
      # ```ruby
      # # typed: true
      # # typed: true
      # # frozen_string_literal: true
      # class Foo; end
      # ```
      #
      # Will be corrected as:
      #
      # ```ruby
      # # typed: true
      # # frozen_string_literal: true
      # class Foo; end
      # ```
      #
      # Other comments or magic comments are left in place.
      class EnforceSingleSigil < ValidSigil
        include RangeHelp

        def investigate(processed_source)
          return if processed_source.tokens.empty?
          sigils = extract_all_sigils(processed_source)
          return unless sigils.size > 1

          sigils[1..sigils.size].each do |token|
            add_offense(token, location: token.pos, message: "Files must only contain one sigil")
          end
        end

        def autocorrect(_node)
          -> (corrector) do
            sigils = extract_all_sigils(processed_source)
            return unless sigils.size > 1

            # The first sigil encountered represents the "real" strictness so remove any below
            sigils[1..sigils.size].each do |token|
              corrector.remove(
                source_range(processed_source.buffer, token.line, (0..token.pos.last_column))
              )
            end
          end
        end

        protected

        def extract_all_sigils(processed_source)
          processed_source.tokens
            .take_while { |token| token.type == :tCOMMENT }
            .find_all { |token| SIGIL_REGEX.match?(token.text) }
        end
      end
    end
  end
end
