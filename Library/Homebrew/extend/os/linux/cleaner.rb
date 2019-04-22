# frozen_string_literal: true

class Cleaner
  private

  def executable_path?(path)
    path.elf? || path.text_executable?
  end
end
