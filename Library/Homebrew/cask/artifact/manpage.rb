# frozen_string_literal: true

require "cask/artifact/moved"

require "extend/hash_validator"
using HashValidator

module Cask
  module Artifact
    class Manpage < Moved
      def self.from_args(cask, *args)
        source_string, section_hash = args
        section = nil

        if section_hash
          raise CaskInvalidError unless section_hash.respond_to?(:keys)

          section_hash.assert_valid_keys!(:section)

          section = section_hash[:section]
        else
          section = source_string.split(".").last
        end

        raise CaskInvalidError, "section should be a positive number" unless section.to_i.positive?

        section_hash ||= {}

        new(cask, source_string, **section_hash)
      end

      def resolve_target(_target)
        config.manpagedir.join("man#{section}", target_name)
      end

      def initialize(cask, source, section: nil)
        @source_section = section

        super(cask, source)
      end

      def extension
        @source.extname.downcase[1..-1].to_s
      end

      def section
        (@source_section || extension).to_i
      end

      def target_name
        "#{@source.basename(@source.extname)}.#{section}"
      end
    end
  end
end
