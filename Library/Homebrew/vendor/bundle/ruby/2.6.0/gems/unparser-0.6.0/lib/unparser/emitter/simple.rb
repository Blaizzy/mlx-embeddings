# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for simple nodes that generate a single token
    class Simple < self
      MAP = {
        __ENCODING__:      '__ENCODING__',
        __FILE__:          '__FILE__',
        __LINE__:          '__LINE__',
        false:             'false',
        forward_arg:       '...',
        forwarded_args:    '...',
        kwnilarg:          '**nil',
        match_nil_pattern: '**nil',
        nil:               'nil',
        redo:              'redo',
        retry:             'retry',
        self:              'self',
        true:              'true',
        zsuper:            'super'
      }.freeze

      handle(*MAP.keys)

    private

      def dispatch
        write(MAP.fetch(node_type))
      end
    end # Simple
  end # Emitter
end # Unparser
