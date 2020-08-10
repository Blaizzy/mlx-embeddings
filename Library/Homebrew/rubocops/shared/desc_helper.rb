# frozen_string_literal: true

require "rubocops/shared/helper_functions"

module RuboCop
  module Cop
    # This module performs common checks the `desc` field in both Formulae and Casks.
    module DescHelper
      include HelperFunctions

      VALID_LOWERCASE_WORDS = %w[
        macOS
      ].freeze

      def audit_desc(type, name, desc_call)
        # Check if a desc is present.
        if desc_call.nil?
          problem "#{type.to_s.capitalize} should have a desc (Description)."
          return
        end

        desc = desc_call.first_argument

        # Check if the desc is empty.
        pure_desc_length = string_content(desc).length
        if pure_desc_length.zero?
          problem "The desc (description) should not be an empty string."
          return
        end

        # Check the desc for leading whitespace.
        problem "Description shouldn't have leading spaces." if regex_match_group(desc, /^\s+/)

        # Check the desc for trailing whitespace.
        problem "Description shouldn't have trailing spaces." if regex_match_group(desc, /\s+$/)

        # Check if "command-line" is spelled incorrectly in the desc.
        if match = regex_match_group(desc, /(command ?line)/i)
          c = match.to_s[0]
          problem "Description should use \"#{c}ommand-line\" instead of \"#{match}\"."
        end

        # Check if the desc starts with "A" or "An".
        if match = regex_match_group(desc, /^(an?)(?=\s)/i)
          problem "Description shouldn't start with an indefinite article, i.e. \"#{match}\"."
        end

        # Check if invalid lowercase words are at the start of a desc.
        if !VALID_LOWERCASE_WORDS.include?(string_content(desc).split.first) &&
           regex_match_group(desc, /^[a-z]/)
          problem "Description should start with a capital letter."
        end

        # Check if the desc starts with the formula's or cask's name.
        problem "Description shouldn't start with the #{type} name." if regex_match_group(desc, /^#{name} /i)

        # Check if a full stop is used at the end of a desc (apart from in the case of "etc.").
        if regex_match_group(desc, /\.$/) && !string_content(desc).end_with?("etc.")
          problem "Description shouldn't end with a full stop."
        end

        # Check if the desc length exceeds 80 characters.
        desc_length = "#{name}: #{string_content(desc)}".length
        max_desc_length = 80
        return if desc_length <= max_desc_length

        problem "Description is too long. \"name: desc\" should be less than #{max_desc_length} characters. " \
                "The current combined length is #{desc_length}."
      end

      def autocorrect_desc(node, name)
        lambda do |corrector|
          /\A(?<quote>["'])(?<correction>.*)(?:\k<quote>)\Z/ =~ node.source

          next if correction.nil?

          correction.gsub!(/^\s+/, "")
          correction.gsub!(/\s+$/, "")

          correction.sub!(/^an?\s+/i, "")

          first_word = correction.split.first
          unless VALID_LOWERCASE_WORDS.include?(first_word)
            first_char = first_word.to_s[0]
            correction[0] = first_char.upcase if first_char
          end

          correction.gsub!(/(ommand ?line)/i, "ommand-line")
          correction.gsub!(/(^|[^a-z])#{name}([^a-z]|$)/i, "\\1\\2")
          correction.gsub!(/^\s+/, "")
          correction.gsub!(/\s+$/, "")
          correction.gsub!(/\.$/, "")

          corrector.replace(node.source_range, "#{quote}#{correction}#{quote}")
        end
      end
    end
  end
end
