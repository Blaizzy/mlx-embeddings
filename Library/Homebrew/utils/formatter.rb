# typed: false
# frozen_string_literal: true

require "utils/tty"

# Helper module for formatting output.
#
# @api private
module Formatter
  module_function

  def arrow(string, color: nil)
    prefix("==>", string, color)
  end

  def headline(string, color: nil)
    arrow("#{Tty.bold}#{string}#{Tty.reset}", color: color)
  end

  def identifier(string)
    "#{Tty.green}#{string}#{Tty.default}"
  end

  def option(string)
    "#{Tty.bold}#{string}#{Tty.reset}"
  end

  def success(string, label: nil)
    label(label, string, :green)
  end

  def warning(string, label: nil)
    label(label, string, :yellow)
  end

  def error(string, label: nil)
    label(label, string, :red)
  end

  # Wraps text to fit within a given number of columns using regular expressions that:
  #
  # 1. convert hard-wrapped paragraphs to a single line
  # 2. add line break and indent to subcommand descriptions
  # 3. find any option descriptions longer than a pre-set length and wrap between words
  #    with a hanging indent, without breaking any words that overflow
  # 4. wrap any remaining description lines that need wrapping with the same indent
  # 5. wrap all lines to the given width.
  #
  # Note that an option (e.g. `--foo`) may not be at the beginning of a line,
  # so we always wrap one word before an option.
  # @see https://github.com/Homebrew/brew/pull/12672
  # @see https://macromates.com/blog/2006/wrapping-text-with-regular-expressions/
  def format_help_text(s, width: 172)
    desc = OPTION_DESC_WIDTH
    indent = width - desc
    s.gsub(/(?<=\S) *\n(?=\S)/, " ")
     .gsub(/([`>)\]]:) /, "\\1\n    ")
     .gsub(/^( +-.+  +(?=\S.{#{desc}}))(.{1,#{desc}})( +|$)(?!-)\n?/, "\\1\\2\n#{" " * indent}")
     .gsub(/^( {#{indent}}(?=\S.{#{desc}}))(.{1,#{desc}})( +|$)(?!-)\n?/, "\\1\\2\n#{" " * indent}")
     .gsub(/(.{1,#{width}})( +|$)(?!-)\n?/, "\\1\n")
  end

  def url(string)
    "#{Tty.underline}#{string}#{Tty.no_underline}"
  end

  def label(label, string, color)
    label = "#{label}:" unless label.nil?
    prefix(label, string, color)
  end
  private_class_method :label

  def prefix(prefix, string, color)
    if prefix.nil? && color.nil?
      string
    elsif prefix.nil?
      "#{Tty.send(color)}#{string}#{Tty.reset}"
    elsif color.nil?
      "#{prefix} #{string}"
    else
      "#{Tty.send(color)}#{prefix}#{Tty.reset} #{string}"
    end
  end
  private_class_method :prefix

  def columns(*objects, gap_size: 2)
    objects = objects.flatten.map(&:to_s)

    fallback = proc do
      return objects.join("\n").concat("\n")
    end

    fallback.call if objects.empty?
    fallback.call if respond_to?(:tty?) ? !tty? : !$stdout.tty?

    console_width = Tty.width
    object_lengths = objects.map { |obj| Tty.strip_ansi(obj).length }
    cols = (console_width + gap_size) / (object_lengths.max + gap_size)

    fallback.call if cols < 2

    rows = (objects.count + cols - 1) / cols
    cols = (objects.count + rows - 1) / rows # avoid empty trailing columns

    col_width = ((console_width + gap_size) / cols) - gap_size

    gap_string = "".rjust(gap_size)

    output = +""

    rows.times do |row_index|
      item_indices_for_row = row_index.step(objects.size - 1, rows).to_a

      first_n = item_indices_for_row[0...-1].map do |index|
        objects[index] + "".rjust(col_width - object_lengths[index])
      end

      # don't add trailing whitespace to last column
      last = objects.values_at(item_indices_for_row.last)

      output.concat((first_n + last)
            .join(gap_string))
            .concat("\n")
    end

    output.freeze
  end
end
