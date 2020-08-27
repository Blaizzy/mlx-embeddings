# frozen_string_literal: true

module PatchELF
  # Helper methods for internal usage.
  module Helper
    # The size of one page.
    PAGE_SIZE = 0x1000

    module_function

    # Color codes for pretty print.
    COLOR_CODE = {
      esc_m: "\e[0m",
      info: "\e[38;5;82m", # light green
      warn: "\e[38;5;230m", # light yellow
      error: "\e[38;5;196m" # heavy red
    }.freeze

    # For wrapping string with color codes for prettier inspect.
    # @param [String] str
    #   Content to colorize.
    # @param [Symbol] type
    #   Specify which kind of color to use, valid symbols are defined in {.COLOR_CODE}.
    # @return [String]
    #   String that wrapped with color codes.
    def colorize(str, type)
      return str unless color_enabled?

      cc = COLOR_CODE
      color = cc.key?(type) ? cc[type] : ''
      "#{color}#{str.sub(COLOR_CODE[:esc_m], color)}#{cc[:esc_m]}"
    end

    # For {#colorize} to decide if need add color codes.
    # @return [Boolean]
    def color_enabled?
      $stderr.tty?
    end

    # @param [Integer] val
    # @param [Integer] align
    # @return [Integer]
    #   Aligned result.
    # @example
    #   aligndown(0x1234)
    #   #=> 4096
    #   aligndown(0x33, 0x20)
    #   #=> 32
    #   aligndown(0x10, 0x8)
    #   #=> 16
    def aligndown(val, align = PAGE_SIZE)
      val - (val & (align - 1))
    end

    # @param [Integer] val
    # @param [Integer] align
    # @return [Integer]
    #   Aligned result.
    # @example
    #   alignup(0x1234)
    #   #=> 8192
    #   alignup(0x33, 0x20)
    #   #=> 64
    #   alignup(0x10, 0x8)
    #   #=> 16
    def alignup(val, align = PAGE_SIZE)
      (val & (align - 1)).zero? ? val : (aligndown(val, align) + align)
    end
  end
end
